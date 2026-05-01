"""
MCP Server for GFM-RAG Knowledge Graph Exploration.

Provides tools:
  1. get_neighbor_entities       - Retrieve neighbor entities of a given entity in the KG.
  2. get_batch_neighbor_entities - Batch version of get_neighbor_entities.
  3. retrieve_documents          - Retrieve documents associated with one or more entities.
  4. add_kg_triplet              - Add a new (head, relation, tail) triplet to the KG.
  5. delete_kg_triplet           - Remove an existing triplet from the KG.
  6. get_kg_edit_log             - Return the log of all KG edits made in this session.

Usage:
    python -m gfmrag.workflow.mcp_server \
        dataset.root=../data/Test/ \
        dataset.data_name=hotpotqa_test \
        graph_retriever.model_path=rmanluo/GFM-RAG-8M

Benchmark mode (no Hydra, requires prior _init_retriever call):
    python -m gfmrag.workflow.mcp_server --benchmark \
        dataset.root=../data/Test/ \
        dataset.data_name=hotpotqa_test \
        graph_retriever.model_path=rmanluo/GFM-RAG-8M
"""

import difflib
import json
import logging
import os
import random
import statistics
import sys
import threading
import time

try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastmcp import FastMCP

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.utils.qa_utils import DocumentRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state – populated by ``_init_retriever`` at server startup
# ---------------------------------------------------------------------------
_qa_data: QADataset | None = None
_id2ent: dict[int, str] | None = None
_ent2id: dict[str, int] | None = None
_rel2id: dict[str, int] | None = None
_id2rel: dict[int, str] | None = None
_edge_index: torch.Tensor | None = None
_edge_type: torch.Tensor | None = None
_doc_retriever: DocumentRetriever | None = None
_ent2docs: torch.Tensor | None = None  # sparse (n_nodes, n_docs)
_entity_names: list[str] | None = None  # all entity names for fuzzy matching

# Adjacency list for O(1) + O(out_degree) neighbor lookup
# node_id → [(rel_type_id, neighbor_id), ...]
_adj_list: dict[int, list[tuple[int, int]]] | None = None

# Ranked neighbor lists from stage1_5 optimization (Algorithm D).
# entity_id → [(rel_name, neighbor_name, score), ...] sorted descending.
# When populated, _get_neighbors_for_node returns neighbors in ranked order.
_ranked_neighbors: dict[int, list[tuple[str, str, float]]] | None = None

# Trigram inverted index for fast fuzzy entity name matching
# trigram → [entity_id, ...]
_trigram_index: dict[str, list[int]] | None = None
_TRIGRAM_MAX_CANDIDATES = 5_000  # cap scored candidates to bound SequenceMatcher time

# Next available IDs for new entities / relations added via KG editing
_next_ent_id: int = 0
_next_rel_id: int = 0

# Thread safety for KG editing tools
_graph_lock = threading.RLock()

# Audit log of all KG edits made during the session
_edit_log: list[dict] = []

DEFAULT_FUZZY_TOP_K = 5  # number of similar entity suggestions to return


# ---------------------------------------------------------------------------
# Fuzzy matching helper
# ---------------------------------------------------------------------------
def _find_similar_entities(query: str, top_k: int = DEFAULT_FUZZY_TOP_K) -> list[dict]:
    """Return the top-k most similar entity names to *query*.

    Scoring backend (in priority order):
    1. rapidfuzz (preferred) – C++ SIMD backend, 10-50× faster than difflib.
       Uses a trigram inverted index to prefilter candidates before scoring.
    2. difflib fallback – trigram prefilter + SequenceMatcher, used only when
       rapidfuzz is not installed.

    The trigram inverted index narrows the candidate set from O(vocab) down to
    at most _TRIGRAM_MAX_CANDIDATES entries before any string-similarity scoring,
    giving an additional 10-100× speedup on large vocabularies.
    """
    assert _entity_names is not None and _id2ent is not None
    query_lower = query.lower().strip()

    # ------------------------------------------------------------------
    # Trigram prefilter (shared by both backends)
    # ------------------------------------------------------------------
    candidates: list[str] | None = None
    if _trigram_index is not None and len(query_lower) >= 2:
        padded = f" {query_lower} "
        overlap: dict[int, int] = {}
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            for eid in _trigram_index.get(tri, []):
                overlap[eid] = overlap.get(eid, 0) + 1

        if overlap:
            top_ids = sorted(overlap, key=overlap.__getitem__, reverse=True)
            top_ids = top_ids[:_TRIGRAM_MAX_CANDIDATES]
            candidates = [_id2ent[eid] for eid in top_ids if eid in _id2ent]

    search_pool = candidates if candidates else _entity_names

    # ------------------------------------------------------------------
    # rapidfuzz path (preferred)
    # ------------------------------------------------------------------
    if _HAS_RAPIDFUZZ:
        hits = _rf_process.extract(
            query_lower, search_pool,
            scorer=_rf_fuzz.WRatio, limit=top_k, score_cutoff=10,
        )
        results = [
            {"entity": name, "similarity": round(score / 100.0, 4)}
            for name, score, _ in hits
        ]
        if results:
            return results
        # If trigram prefilter found no good match, retry on full vocab
        if candidates is not None:
            hits = _rf_process.extract(
                query_lower, _entity_names,
                scorer=_rf_fuzz.WRatio, limit=top_k, score_cutoff=10,
            )
            return [
                {"entity": name, "similarity": round(score / 100.0, 4)}
                for name, score, _ in hits
            ]
        return []

    # ------------------------------------------------------------------
    # difflib fallback (no rapidfuzz installed)
    # ------------------------------------------------------------------
    if candidates:
        scored = [
            (name, difflib.SequenceMatcher(None, query_lower, name).ratio())
            for name in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [
            {"entity": name, "similarity": round(score, 4)}
            for name, score in scored[:top_k]
            if score > 0.1
        ]
        if results:
            return results

    close = difflib.get_close_matches(query_lower, _entity_names, n=top_k, cutoff=0.4)
    if not close:
        close = difflib.get_close_matches(query_lower, _entity_names, n=top_k, cutoff=0.1)
    scored = [
        (name, difflib.SequenceMatcher(None, query_lower, name).ratio())
        for name in close
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"entity": name, "similarity": round(score, 4)} for name, score in scored[:top_k]]


# ---------------------------------------------------------------------------
# Internal neighbor lookup (no MCP decoration, reused by batch tool)
# ---------------------------------------------------------------------------
def _get_neighbors_for_node(
    node_id: int,
    relation_type: str | None = None,
) -> dict[str, list[str]]:
    """Return {relation_name: [neighbor_name, ...]} for *node_id*.

    When ranked neighbor lists are available (from stage1_5 optimization),
    neighbors within each relation group are returned in confidence-descending
    order.  Otherwise falls back to the raw adjacency list.
    """
    assert _adj_list is not None and _id2rel is not None and _id2ent is not None

    result: dict[str, list[str]] = {}

    if _ranked_neighbors is not None and node_id in _ranked_neighbors:
        # Use pre-sorted ranked lists from Algorithm D
        for rel_name, nbr_name, _score in _ranked_neighbors[node_id]:
            result.setdefault(rel_name, []).append(nbr_name)
    else:
        for rel_id, nbr_id in _adj_list.get(node_id, []):
            rel_name = _id2rel.get(rel_id, f"relation_{rel_id}")
            ent_name = _id2ent.get(nbr_id, f"entity_{nbr_id}")
            result.setdefault(rel_name, []).append(ent_name)

    if relation_type is not None:
        rel_key = relation_type.lower().strip()
        result = {k: v for k, v in result.items() if k.lower() == rel_key}

    return result


# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="GFM-RAG Knowledge Graph Server",
    instructions=(
        "This server exposes a knowledge graph built by GFM-RAG. "
        "You can query neighbor entities of a given entity or retrieve "
        "documents associated with an entity. You can also edit the KG by "
        "adding or deleting triplets."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 – get_neighbor_entities
# ---------------------------------------------------------------------------
@mcp.tool()
def get_neighbor_entities(
    entity_name: str,
    relation_type: str | None = None,
) -> str:
    """Retrieve the neighbor entities of a given entity in the knowledge graph.

    Args:
        entity_name: The name of the entity to look up (case-insensitive).
        relation_type: Optional relation type to filter neighbors. If not
            provided, all neighbors (across every relation) are returned.

    Returns:
        A JSON string with the neighbor information.  The top-level keys are
        relation names; each value is a list of neighboring entity names.
        If the entity is not found, an error message is returned.
    """
    assert _ent2id is not None, "Server not initialised"
    assert _adj_list is not None

    entity_key = entity_name.lower().strip()
    if entity_key not in _ent2id:
        # print(f"Entity '{entity_name}' not found. Attempting fuzzy match …")
        suggestions = _find_similar_entities(entity_key)
        # print(f"Top {len(suggestions)} similar entities: {suggestions}")
        return json.dumps(
            {
                "error": f"Entity '{entity_name}' not found in the knowledge graph.",
                "similar_entities": suggestions,
                "hint": "Try calling the tool again with one of the suggested entity names.",
            }
        )

    node_id = _ent2id[entity_key]
    result = _get_neighbors_for_node(node_id, relation_type)

    if relation_type is not None and not result:
        all_neighbors = _get_neighbors_for_node(node_id)
        return json.dumps(
            {
                "error": (
                    f"No neighbors found for entity '{entity_name}' "
                    f"with relation type '{relation_type}'."
                ),
                "available_relations": list(all_neighbors.keys()),
            }
        )

    return json.dumps(
        {
            "entity": entity_name,
            "neighbors": result,
            "total_neighbor_count": sum(len(v) for v in result.values()),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Tool 2 – get_batch_neighbor_entities
# ---------------------------------------------------------------------------
@mcp.tool()
def get_batch_neighbor_entities(
    entity_names: list[str],
    relation_type: str | None = None,
) -> str:
    """Retrieve neighbor entities for multiple entities in a single call.

    Equivalent to calling get_neighbor_entities for each entity individually,
    but avoids the per-call overhead when several entities are known upfront
    (e.g., during multi-hop reasoning).

    Args:
        entity_names: List of entity names to look up (case-insensitive).
        relation_type: Optional relation type filter applied to all entities.

    Returns:
        A JSON object mapping each input entity name to its neighbor dict
        (same format as get_neighbor_entities), plus a ``not_found`` list
        for any entities that could not be resolved.
    """
    assert _ent2id is not None, "Server not initialised"
    assert _adj_list is not None

    results: dict[str, dict] = {}
    not_found: list[dict] = []

    for name in entity_names:
        entity_key = name.lower().strip()
        if entity_key not in _ent2id:
            suggestions = _find_similar_entities(entity_key)
            not_found.append({"entity": name, "similar_entities": suggestions})
            continue

        node_id = _ent2id[entity_key]
        neighbors = _get_neighbors_for_node(node_id, relation_type)
        results[name] = {
            "neighbors": neighbors,
            "total_neighbor_count": sum(len(v) for v in neighbors.values()),
        }

    return json.dumps(
        {"entities": results, "not_found": not_found},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Tool 3 – retrieve_documents
# ---------------------------------------------------------------------------
@mcp.tool()
def retrieve_documents(
    entity_names: list[str],
    top_k: int = 5,
) -> str:
    """Retrieve documents associated with one or more entities in the knowledge graph.

    The entity-to-document mapping is derived from the GFM-RAG dataset.  When
    multiple entities are provided their document scores are aggregated so that
    documents relevant to *all* of them are ranked highest.

    Args:
        entity_names: A list of entity names to look up (case-insensitive).
        top_k: Maximum number of documents to return (default 5).

    Returns:
        A JSON string with the list of retrieved documents. Each document has
        keys: ``title``, ``content``, ``score``, ``norm_score``.
        Entities that are not found are listed under ``not_found`` together
        with fuzzy-match suggestions.
    """
    assert _ent2id is not None and _ent2docs is not None
    assert _doc_retriever is not None

    num_nodes = _ent2docs.shape[0]
    ent_pred = torch.zeros(1, num_nodes, device=_ent2docs.device)

    resolved_entities: list[str] = []
    not_found: list[dict] = []

    for name in entity_names:
        entity_key = name.lower().strip()
        if entity_key in _ent2id:
            node_id = _ent2id[entity_key]
            ent_pred[0, node_id] = 1.0
            resolved_entities.append(entity_key)
        else:
            # print(f"Entity '{name}' not found. Attempting fuzzy match …")
            suggestions = _find_similar_entities(entity_key)
            # print(f"Top {len(suggestions)} similar entities: {suggestions}")
            not_found.append(
                {
                    "entity": name,
                    "similar_entities": suggestions,
                }
            )

    # If none of the requested entities were found, return early with suggestions
    if not resolved_entities:
        return json.dumps(
            {
                "error": "None of the provided entities were found in the knowledge graph.",
                "not_found": not_found,
                "hint": "Try calling the tool again with corrected entity names.",
            }
        )

    doc_scores = torch.sparse.mm(ent_pred, _ent2docs)[0]  # (n_docs,)

    # Retrieve the top-k documents using the standard DocumentRetriever
    retrieved_docs = _doc_retriever(doc_scores.cpu(), top_k=top_k)

    result: dict = {
        "entities": resolved_entities,
        "documents": retrieved_docs,
        "total_returned": len(retrieved_docs),
    }
    if not_found:
        result["not_found"] = not_found
        result["hint"] = (
            "Some entities were not found. Documents were retrieved using "
            "the resolved entities only. Consider retrying with corrected names."
        )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4 – add_kg_triplet  (KG editing)
# ---------------------------------------------------------------------------
@mcp.tool()
def add_kg_triplet(head: str, relation: str, tail: str) -> str:
    """Add a new (head, relation, tail) triplet to the knowledge graph.

    If the head or tail entity does not yet exist in the KG it is created.
    If the relation does not exist it is created.  The edit is recorded in
    the session edit log and is visible to subsequent get_neighbor_entities
    calls immediately.

    Note: only the forward edge (head → tail) is added.  If you also need
    the reverse direction, call this tool again with head and tail swapped
    and the appropriate inverse relation name.

    Args:
        head: Name of the source entity (case-insensitive).
        relation: Relation / edge label (case-insensitive).
        tail: Name of the target entity (case-insensitive).

    Returns:
        A JSON confirmation string.
    """
    global _next_ent_id, _next_rel_id, _edit_log

    assert _ent2id is not None and _id2ent is not None
    assert _rel2id is not None and _id2rel is not None
    assert _adj_list is not None and _entity_names is not None

    head_key = head.lower().strip()
    tail_key = tail.lower().strip()
    rel_key = relation.lower().strip()

    with _graph_lock:
        # Resolve or create entity IDs
        if head_key not in _ent2id:
            _ent2id[head_key] = _next_ent_id
            _id2ent[_next_ent_id] = head_key
            _entity_names.append(head_key)
            if _trigram_index is not None:
                padded = f" {head_key} "
                for i in range(len(padded) - 2):
                    _trigram_index.setdefault(padded[i : i + 3], []).append(_next_ent_id)
            _next_ent_id += 1
            head_created = True
        else:
            head_created = False

        if tail_key not in _ent2id:
            _ent2id[tail_key] = _next_ent_id
            _id2ent[_next_ent_id] = tail_key
            _entity_names.append(tail_key)
            if _trigram_index is not None:
                padded = f" {tail_key} "
                for i in range(len(padded) - 2):
                    _trigram_index.setdefault(padded[i : i + 3], []).append(_next_ent_id)
            _next_ent_id += 1
            tail_created = True
        else:
            tail_created = False

        # Resolve or create relation ID
        if rel_key not in _rel2id:
            _rel2id[rel_key] = _next_rel_id
            _id2rel[_next_rel_id] = rel_key
            _next_rel_id += 1
            rel_created = True
        else:
            rel_created = False

        head_id = _ent2id[head_key]
        tail_id = _ent2id[tail_key]
        rel_id = _rel2id[rel_key]

        # Check for duplicate edge
        existing = _adj_list.get(head_id, [])
        if (rel_id, tail_id) in existing:
            return json.dumps(
                {
                    "status": "skipped",
                    "reason": "Triplet already exists.",
                    "triplet": {"head": head_key, "relation": rel_key, "tail": tail_key},
                }
            )

        # Add the forward edge
        _adj_list.setdefault(head_id, []).append((rel_id, tail_id))

        # Record the edit
        entry = {
            "op": "add",
            "head": head_key,
            "relation": rel_key,
            "tail": tail_key,
            "head_id": head_id,
            "rel_id": rel_id,
            "tail_id": tail_id,
            "head_created": head_created,
            "tail_created": tail_created,
            "rel_created": rel_created,
        }
        _edit_log.append(entry)

    return json.dumps(
        {
            "status": "added",
            "triplet": {"head": head_key, "relation": rel_key, "tail": tail_key},
            "new_entities": [e for e, c in [(head_key, head_created), (tail_key, tail_created)] if c],
            "new_relations": [rel_key] if rel_created else [],
        }
    )


# ---------------------------------------------------------------------------
# Tool 5 – delete_kg_triplet  (KG editing)
# ---------------------------------------------------------------------------
@mcp.tool()
def delete_kg_triplet(head: str, relation: str, tail: str) -> str:
    """Remove an existing (head, relation, tail) triplet from the knowledge graph.

    Only the exact forward edge is removed.  If the graph also contains a
    reverse edge it must be deleted separately.

    Args:
        head: Name of the source entity (case-insensitive).
        relation: Relation / edge label (case-insensitive).
        tail: Name of the target entity (case-insensitive).

    Returns:
        A JSON confirmation string.
    """
    global _edit_log

    assert _ent2id is not None and _rel2id is not None and _adj_list is not None

    head_key = head.lower().strip()
    tail_key = tail.lower().strip()
    rel_key = relation.lower().strip()

    with _graph_lock:
        if head_key not in _ent2id or tail_key not in _ent2id or rel_key not in _rel2id:
            missing = [
                x
                for x, d in [(head_key, _ent2id), (tail_key, _ent2id), (rel_key, _rel2id)]
                if x not in d
            ]
            return json.dumps(
                {
                    "status": "not_found",
                    "reason": f"Unknown identifiers: {missing}",
                    "triplet": {"head": head_key, "relation": rel_key, "tail": tail_key},
                }
            )

        head_id = _ent2id[head_key]
        tail_id = _ent2id[tail_key]
        rel_id = _rel2id[rel_key]

        edges = _adj_list.get(head_id, [])
        edge = (rel_id, tail_id)
        if edge not in edges:
            return json.dumps(
                {
                    "status": "not_found",
                    "reason": "Edge does not exist in the adjacency list.",
                    "triplet": {"head": head_key, "relation": rel_key, "tail": tail_key},
                }
            )

        edges.remove(edge)

        entry = {
            "op": "delete",
            "head": head_key,
            "relation": rel_key,
            "tail": tail_key,
        }
        _edit_log.append(entry)

    return json.dumps(
        {
            "status": "deleted",
            "triplet": {"head": head_key, "relation": rel_key, "tail": tail_key},
        }
    )


# ---------------------------------------------------------------------------
# Tool 6 – get_kg_edit_log
# ---------------------------------------------------------------------------
@mcp.tool()
def get_kg_edit_log() -> str:
    """Return the full log of KG edits made in the current session.

    Returns:
        A JSON array of edit records.  Each record has keys: ``op``
        (``"add"`` or ``"delete"``), ``head``, ``relation``, ``tail``.
    """
    return json.dumps(_edit_log, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Initialisation – load dataset & build look-up structures
# ---------------------------------------------------------------------------
def _init_retriever(cfg: DictConfig) -> None:
    """Load the QADataset and prepare all look-up structures."""

    global _qa_data, _id2ent, _ent2id, _rel2id, _id2rel
    global _edge_index, _edge_type, _doc_retriever, _ent2docs, _entity_names
    global _adj_list, _trigram_index, _next_ent_id, _next_rel_id, _edit_log
    global _ranked_neighbors

    logger.info("Loading GFM-RAG model and dataset …")
    print(cfg.graph_retriever.model_path)
    _, model_config = utils.load_model_from_pretrained(cfg.graph_retriever.model_path)

    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )

    # Use GPU for sparse MM in retrieve_documents if available; adj-list
    # neighbor lookups are pure Python and stay on CPU regardless.
    retrieval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Retrieval device: %s", retrieval_device)

    kg = qa_data.kg.to("cpu")  # edge tensors only used to build adj list
    ent2docs = qa_data.ent2docs.to(retrieval_device)

    _qa_data = qa_data
    _ent2id = qa_data.ent2id  # {entity_name -> int}
    _id2ent = {v: k for k, v in _ent2id.items()}
    _rel2id = qa_data.rel2id  # {relation_name -> int}
    _id2rel = {v: k for k, v in _rel2id.items()}
    _edge_index = kg.edge_index  # (2, num_edges)
    _edge_type = kg.edge_type  # (num_edges,)
    _ent2docs = ent2docs  # sparse (n_nodes, n_docs)
    _doc_retriever = DocumentRetriever(qa_data.doc, qa_data.id2doc)
    _entity_names = list(_ent2id.keys())  # pre-built list for fuzzy matching

    # ------------------------------------------------------------------
    # Build adjacency list: O(num_edges) once → O(out_degree) per query
    # ------------------------------------------------------------------
    logger.info("Building adjacency list …")
    t0 = time.time()
    adj: dict[int, list[tuple[int, int]]] = {}
    srcs = _edge_index[0].tolist()
    tgts = _edge_index[1].tolist()
    rels = _edge_type.tolist()
    for src, tgt, rel in zip(srcs, tgts, rels):
        adj.setdefault(src, []).append((rel, tgt))
    _adj_list = adj
    logger.info(
        "Adjacency list built in %.2fs – %d nodes with edges",
        time.time() - t0,
        len(_adj_list),
    )

    # ------------------------------------------------------------------
    # Build trigram inverted index for fast fuzzy entity name matching
    # ------------------------------------------------------------------
    logger.info("Building trigram index …")
    t0 = time.time()
    tri_idx: dict[str, list[int]] = {}
    for name, ent_id in _ent2id.items():
        padded = f" {name} "
        for i in range(len(padded) - 2):
            tri = padded[i : i + 3]
            tri_idx.setdefault(tri, []).append(ent_id)
    _trigram_index = tri_idx
    logger.info(
        "Trigram index built in %.2fs – %d unique trigrams",
        time.time() - t0,
        len(_trigram_index),
    )

    # ------------------------------------------------------------------
    # Load stage1_5 optimization artifacts if available
    # ------------------------------------------------------------------
    raw_stage = getattr(cfg.dataset, "raw_stage", "stage1")
    stage_dir = os.path.join(cfg.dataset.root, cfg.dataset.data_name, "processed", raw_stage)

    # Alias table: add aliases as exact-match expansions in _ent2id
    alias_path = os.path.join(stage_dir, "alias_table.json")
    if os.path.exists(alias_path):
        with open(alias_path, encoding="utf-8") as f:
            alias_table: dict[str, str] = json.load(f)
        n_aliases_added = 0
        for alias, canonical in alias_table.items():
            alias_lower = alias.lower().strip()
            canonical_lower = canonical.lower().strip()
            if alias_lower not in _ent2id and canonical_lower in _ent2id:
                canonical_id = _ent2id[canonical_lower]
                _ent2id[alias_lower] = canonical_id
                _entity_names.append(alias_lower)
                # Update trigram index for the new alias
                if _trigram_index is not None:
                    padded = f" {alias_lower} "
                    for i in range(len(padded) - 2):
                        _trigram_index.setdefault(padded[i : i + 3], []).append(canonical_id)
                n_aliases_added += 1
        logger.info(
            "Alias table loaded: %d aliases → %d added to entity index",
            len(alias_table),
            n_aliases_added,
        )

    # Ranked neighbor lists: replace adjacency ordering with confidence-sorted lists
    rankings_path = os.path.join(stage_dir, "neighbor_rankings.json")
    if os.path.exists(rankings_path):
        with open(rankings_path, encoding="utf-8") as f:
            raw_rankings: dict[str, list] = json.load(f)
        # Convert entity names → entity IDs for fast lookup
        ranked: dict[int, list[tuple[str, str, float]]] = {}
        for ent_name, neighbors in raw_rankings.items():
            ent_key = ent_name.lower().strip()
            if ent_key in _ent2id:
                eid = _ent2id[ent_key]
                # neighbors: [[rel, nbr, score], ...]
                ranked[eid] = [(r, n, float(s)) for r, n, s in neighbors]
        _ranked_neighbors = ranked
        logger.info(
            "Ranked neighbor lists loaded: %d entities have pre-sorted neighbors",
            len(_ranked_neighbors),
        )

    # IDs for new entities / relations created via KG editing
    _next_ent_id = len(_ent2id)
    _next_rel_id = len(_rel2id)
    _edit_log = []

    logger.info(
        "MCP server ready – %d entities, %d relations, %d documents (retrieval on %s)",
        len(_ent2id),
        len(_rel2id),
        len(qa_data.doc),
        retrieval_device,
    )


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------
def benchmark_kg_retrieval(n_queries: int = 1000) -> None:
    """Benchmark get_neighbor_entities and retrieve_documents throughput.

    Prints mean / p95 / p99 latencies and overall queries-per-second.
    Must be called after _init_retriever.
    """
    assert _entity_names is not None and len(_entity_names) > 0, "Retriever not initialised"

    sample_entities = random.choices(_entity_names, k=n_queries)

    # --- get_neighbor_entities ---
    timings_nbr: list[float] = []
    for ent in sample_entities:
        t0 = time.perf_counter()
        get_neighbor_entities(ent)
        timings_nbr.append(time.perf_counter() - t0)

    # --- retrieve_documents (single entity each) ---
    timings_doc: list[float] = []
    for ent in sample_entities:
        t0 = time.perf_counter()
        retrieve_documents([ent], top_k=3)
        timings_doc.append(time.perf_counter() - t0)

    def _stats(label: str, timings: list[float]) -> None:
        t_ms = [t * 1000 for t in timings]
        t_sorted = sorted(t_ms)
        p95 = t_sorted[int(len(t_sorted) * 0.95)]
        p99 = t_sorted[int(len(t_sorted) * 0.99)]
        qps = len(timings) / sum(timings)
        print(
            f"{label}: mean={statistics.mean(t_ms):.2f}ms  "
            f"p95={p95:.2f}ms  p99={p99:.2f}ms  "
            f"qps={qps:.1f}"
        )

    print(f"\n=== KG Retrieval Benchmark (n={n_queries}) ===")
    _stats("get_neighbor_entities", timings_nbr)
    _stats("retrieve_documents   ", timings_doc)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def main() -> None:
    """Launch the MCP server.

    Hydra config overrides can be passed as CLI arguments, e.g.:

        python -m gfmrag.workflow.mcp_server \
            dataset.data_name=hotpotqa_test \
            graph_retriever.model_path=rmanluo/GFM-RAG-8M

    Add --benchmark to run the throughput benchmark instead of starting the server.
    """
    run_benchmark = "--benchmark" in sys.argv
    overrides = [a for a in sys.argv[1:] if a != "--benchmark"]

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="stage3_qa_inference", overrides=overrides)

    logging.basicConfig(level=logging.INFO)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    _init_retriever(cfg)

    if run_benchmark:
        n = int(os.environ.get("BENCHMARK_N", "1000"))
        benchmark_kg_retrieval(n_queries=n)
        return

    # Run the MCP server (stdio transport by default)
    mcp.run()


if __name__ == "__main__":
    main()
