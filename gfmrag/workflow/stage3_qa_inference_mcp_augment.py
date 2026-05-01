import json
import logging
import os
import re
import threading
from multiprocessing.dummy import Pool as ThreadPool

import dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from tqdm import tqdm

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.ultra import query_utils

# Access mcp_server module globals for the KG revert operation
import gfmrag.workflow.mcp_server as _mcp_module
from gfmrag.workflow.mcp_server import _init_retriever

# Re-use the agent loop and tool definitions from the base MCP inference module
from gfmrag.workflow.stage3_qa_inference_mcp import (
    _BASE_TOOL_DISPATCH,
    _BASE_TOOL_SCHEMAS,
    _KG_EDIT_TOOL_DISPATCH,
    _KG_EDIT_TOOL_SCHEMAS,
    _SYSTEM_PROMPT_BASE,
    agent_qa,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch QA – system-prompt addition
# ---------------------------------------------------------------------------
_BATCH_SYSTEM_PROMPT_SUFFIX = """\
You will be given a numbered batch of questions.  Use tools freely across all \
questions in the session – entities and evidence found for one question may \
help answer others.

After you have gathered sufficient evidence for ALL questions, output your \
final answers in this exact block (one line per question, no other text after \
the block):

Q1 Evidence: <reasoning path> Answer: <concise answer>
Q2 Evidence: <reasoning path> Answer: <concise answer>
...

Do not emit partial answers mid-session.  Wait until every question is \
resolved before writing the final block.
"""

# ---------------------------------------------------------------------------
# Augment QA – system prompts
# ---------------------------------------------------------------------------
_AUGMENT_EDIT_SYSTEM_PROMPT = """\
You are a knowledge graph engineer.  A retrieval agent has already answered a \
question; its reasoning evidence and final answer are provided to you as context.

Your task is to enrich the knowledge graph by calling add_kg_triplet for every \
key fact that was decisive in reaching the answer.  Specifically add:
  1. Direct answer facts: the (subject, relation, answer_entity) edge(s) that \
directly encode the answer.
  2. Bridge facts: intermediate (entity, relation, entity) edges that connected \
the question entity to the answer entity across hops.

Rules:
  • Use lowercase, canonical entity/relation names (e.g. "place_of_birth").
  • Only add facts *explicitly* present in the provided evidence.  Do not \
retrieve additional information and do not speculate.
  • If a fact already exists the tool will report "skipped" – that is fine.
  • Call get_kg_edit_log at the end to confirm what was added.
  • When you are done, output a brief summary of the triplets you added.
"""


def _build_batch_system_prompt() -> str:
    return _SYSTEM_PROMPT_BASE + _BATCH_SYSTEM_PROMPT_SUFFIX


def _build_augment_baseline_system_prompt() -> str:
    """Pass 1 – standard read-only prompt."""
    return _SYSTEM_PROMPT_BASE


def _build_augment_enriched_system_prompt() -> str:
    """Pass 3 – read-only on the now-enriched graph."""
    note = (
        "Note: the knowledge graph has been pre-enriched with additional facts. "
        "You may find that direct answers are now reachable via graph traversal "
        "with fewer hops than usual.\n\n"
    )
    return _SYSTEM_PROMPT_BASE + note


# ---------------------------------------------------------------------------
# Pass 2: KG-edit agent seeded with Pass 1 evidence (no retrieval)
# ---------------------------------------------------------------------------
def _augment_from_pass1(
    client: OpenAI,
    model_name: str,
    question: str,
    pass1_response: str,
    temperature: float,
    max_steps: int,
) -> tuple[list[dict], dict]:
    """Pass 2: KG-edit agent loop seeded with Pass 1 output.

    The agent receives the question and Pass 1's evidence + answer as its
    user message.  Only KG-edit tools are available (add_kg_triplet,
    delete_kg_triplet, get_kg_edit_log) – no retrieval tools – so the agent
    derives all triplets from the supplied context without re-querying the
    graph or document corpus.

    The edit log is assumed to have been cleared by the caller immediately
    before this call.

    Returns:
        (edits, token_statistics)
        ``edits`` is the list of KG edit records produced in this call.
    """
    user_content = (
        f"Question: {question}\n\n"
        f"Evidence and answer from the retrieval agent:\n{pass1_response}"
    )

    result = agent_qa(
        client=client,
        model_name=model_name,
        question=user_content,
        tool_schemas=list(_KG_EDIT_TOOL_SCHEMAS),
        tool_dispatch=dict(_KG_EDIT_TOOL_DISPATCH),
        system_prompt=_AUGMENT_EDIT_SYSTEM_PROMPT,
        max_steps=max_steps,
        temperature=temperature,
        question_entities=None,
    )

    edits = list(_mcp_module._edit_log)
    return edits, result["token_statistics"]


# ---------------------------------------------------------------------------
# KG revert helper
# ---------------------------------------------------------------------------
def _revert_kg_edits(edits: list[dict], ent_count_before: int, rel_count_before: int) -> None:
    """Undo every edit in *edits* (a snapshot of ``_mcp_module._edit_log``).

    Must be called from a single thread; acquires ``_graph_lock`` internally.
    Processes edits in reverse chronological order so multi-hop dependency
    chains are unwound safely.

    Args:
        edits: List of edit records as produced by add_kg_triplet /
            delete_kg_triplet, in the order they were applied.
        ent_count_before: Value of ``_mcp_module._next_ent_id`` before the
            edits were applied.  Used to restore the ID counter.
        rel_count_before: Value of ``_mcp_module._next_rel_id`` before the
            edits were applied.
    """
    if not edits:
        return

    with _mcp_module._graph_lock:
        for edit in reversed(edits):
            op = edit["op"]

            if op == "add":
                head_id = edit["head_id"]
                rel_id = edit["rel_id"]
                tail_id = edit["tail_id"]

                # Remove the forward edge
                edges = _mcp_module._adj_list.get(head_id, [])
                edge = (rel_id, tail_id)
                if edge in edges:
                    edges.remove(edge)
                if not edges and head_id in _mcp_module._adj_list:
                    del _mcp_module._adj_list[head_id]

                # Remove newly created entities (head, tail)
                for ent_key, ent_id, was_created in [
                    (edit["head"], head_id, edit.get("head_created", False)),
                    (edit["tail"], tail_id, edit.get("tail_created", False)),
                ]:
                    if not was_created:
                        continue
                    _mcp_module._ent2id.pop(ent_key, None)
                    _mcp_module._id2ent.pop(ent_id, None)
                    try:
                        _mcp_module._entity_names.remove(ent_key)
                    except ValueError:
                        pass
                    # Remove entity from trigram index
                    if _mcp_module._trigram_index is not None:
                        padded = f" {ent_key} "
                        for i in range(len(padded) - 2):
                            tri = padded[i : i + 3]
                            tri_list = _mcp_module._trigram_index.get(tri)
                            if tri_list and ent_id in tri_list:
                                tri_list.remove(ent_id)

                # Remove newly created relation
                if edit.get("rel_created", False):
                    _mcp_module._rel2id.pop(edit["relation"], None)
                    _mcp_module._id2rel.pop(rel_id, None)

            elif op == "delete":
                # Re-insert the deleted edge (look up current IDs)
                head_key = edit["head"]
                tail_key = edit["tail"]
                rel_key = edit["relation"]
                ent2id = _mcp_module._ent2id
                rel2id = _mcp_module._rel2id
                if head_key in ent2id and tail_key in ent2id and rel_key in rel2id:
                    h_id = ent2id[head_key]
                    t_id = ent2id[tail_key]
                    r_id = rel2id[rel_key]
                    _mcp_module._adj_list.setdefault(h_id, []).append((r_id, t_id))

        # Restore ID counters
        _mcp_module._next_ent_id = ent_count_before
        _mcp_module._next_rel_id = rel_count_before

        # Clear the session edit log
        _mcp_module._edit_log.clear()


# ---------------------------------------------------------------------------
# Batch QA helpers
# ---------------------------------------------------------------------------
def _build_batch_user_message(samples: list[dict]) -> tuple[str, list[str]]:
    """Build a single user message covering all questions in a batch.

    Returns:
        (user_content, all_entity_hints) where ``all_entity_hints`` is the
        union of question entities across all samples.
    """
    lines = [f"You must answer all {len(samples)} questions below.\n"]
    all_entities: list[str] = []

    for i, s in enumerate(samples, 1):
        lines.append(f"Q{i}: {s['question']}")
        entities = s.get("question_entities", [])
        if entities:
            lines.append(f"     Entities: {', '.join(entities)}")
            all_entities.extend(entities)
        lines.append("")

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_entities = [e for e in all_entities if not (e in seen or seen.add(e))]  # type: ignore[func-returns-value]

    return "\n".join(lines), unique_entities


_Q_ANSWER_RE = re.compile(
    r"Q(\d+)[^\n]*?Answer\s*:\s*(.+?)(?=\nQ\d+|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_batch_answers(response: str, n: int) -> list[str]:
    """Extract per-question answers from a batch response string.

    Tries structured ``Q{i} ... Answer: ...`` format first, then falls back
    to splitting on question-number markers.  Returns a list of *n* strings
    (empty string where parsing fails).
    """
    answers: dict[int, str] = {}

    for m in _Q_ANSWER_RE.finditer(response):
        idx = int(m.group(1))
        answer_text = m.group(2).strip().splitlines()[0].strip()
        answers[idx] = answer_text

    # Fallback: if fewer than n answers parsed, try a looser split
    if len(answers) < n:
        loose_re = re.compile(r"(?:^|\n)\s*(?:Q|Question\s*)(\d+)[^\n]*\n(.*?)(?=\n\s*(?:Q|Question\s*)\d+|\Z)", re.DOTALL)
        for m in loose_re.finditer(response):
            idx = int(m.group(1))
            if idx not in answers:
                chunk = m.group(2).strip()
                # grab the last "Answer:" line in the chunk
                ans_m = re.search(r"Answer\s*:\s*(.+)", chunk, re.IGNORECASE)
                if ans_m:
                    answers[idx] = ans_m.group(1).strip().splitlines()[0].strip()

    return [answers.get(i, "") for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Mode 1 – Batch QA
# ---------------------------------------------------------------------------
def run_batch_qa(
    client: OpenAI,
    model_name: str,
    samples: list[dict],
    max_steps: int,
    temperature: float,
    system_prompt: str,
) -> list[dict]:
    """Run the agent on a batch of questions in a single session.

    Returns a list of result dicts (one per sample in the batch), each with
    the same schema as the single-question path plus a ``batch_id`` field.
    """
    batch_id = "_".join(s["id"] for s in samples[:3])
    if len(samples) > 3:
        batch_id += f"_+{len(samples) - 3}"

    user_message, all_entities = _build_batch_user_message(samples)

    tool_schemas = list(_BASE_TOOL_SCHEMAS)
    tool_dispatch = dict(_BASE_TOOL_DISPATCH)

    agent_result = agent_qa(
        client=client,
        model_name=model_name,
        question=user_message,
        tool_schemas=tool_schemas,
        tool_dispatch=tool_dispatch,
        system_prompt=system_prompt,
        max_steps=max_steps,
        temperature=temperature,
        question_entities=all_entities,
    )

    responses = _parse_batch_answers(agent_result["response"], len(samples))
    n_tool_calls = len(agent_result["tool_calls"])
    n_steps = len(agent_result["logs"])

    results = []
    for i, sample in enumerate(samples):
        results.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_aliases": sample.get("answer_aliases", []),
                "response": responses[i],
                "batch_id": batch_id,
                "batch_size": len(samples),
                "batch_position": i + 1,
                # Shared across the batch; included per-record for evaluator compatibility
                "full_batch_response": agent_result["response"],
                "logs": agent_result["logs"],
                "tool_calls": agent_result["tool_calls"],
                "token_statistics": agent_result["token_statistics"],
                "n_steps": n_steps,
                "n_tool_calls": n_tool_calls,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Mode 2 – Augmentation QA
# ---------------------------------------------------------------------------
def run_augment_qa(
    client: OpenAI,
    model_name: str,
    sample: dict,
    max_steps: int,
    temperature: float,
    pass1_cache: dict[str, dict] | None = None,
) -> dict:
    """Three-pass evaluation of a single sample.

    Pass 1 (baseline) – read-only agent answers the question; establishes the
                        reference token / tool-call cost.
    Pass 2 (augment)  – single LLM call (no tools) that reads the question and
                        Pass 1's evidence + answer, extracts key facts as
                        (head, relation, tail) triplets, and adds them to the
                        KG.  overhead = pass2.tokens
    Pass 3 (enriched) – same question on the now-enriched graph, read-only.
                        efficiency_gain = pass3.tokens - pass1.tokens

    KG edits from Pass 2 are reverted after Pass 3.
    """
    sid = sample["id"]
    question = sample["question"]
    question_entities = sample.get("question_entities", [])

    readonly_schemas = list(_BASE_TOOL_SCHEMAS)
    readonly_dispatch = dict(_BASE_TOOL_DISPATCH)

    # --- Pass 1: baseline (no KG editing) -----------------------------------
    if pass1_cache and sid in pass1_cache:
        logger.debug("[%s] Pass 1 loaded from cache", sid)
        pass1_result = pass1_cache[sid]
    else:
        logger.debug("[%s] Pass 1 baseline …", sid)
        pass1_result = agent_qa(
            client=client,
            model_name=model_name,
            question=question,
            tool_schemas=readonly_schemas,
            tool_dispatch=readonly_dispatch,
            system_prompt=_build_augment_baseline_system_prompt(),
            max_steps=max_steps,
            temperature=temperature,
            question_entities=question_entities,
        )
    logger.debug(
        "[%s] Pass 1 done – %d steps, %d tool calls, %d tokens",
        sid,
        len(pass1_result["logs"]),
        len(pass1_result["tool_calls"]),
        pass1_result["token_statistics"]["total_tokens"],
    )

    def _summarise_agent(result: dict) -> dict:
        return {
            "response": result["response"],
            "n_steps": len(result["logs"]),
            "n_tool_calls": len(result["tool_calls"]),
            "token_statistics": result["token_statistics"],
            "logs": result["logs"],
            "tool_calls": result["tool_calls"],
        }

    # --- Skip augmentation if Pass 1 converged quickly ----------------------
    num_pass1_steps = len(pass1_result["logs"])
    if num_pass1_steps <= 5 or not pass1_result.get("response") or "Answer" not in pass1_result["response"].strip() or "Insufficient" in pass1_result["response"].strip():
        logger.debug(
            "[%s] Pass 1 used only %d steps or has no response – skipping augmentation",
            sid, num_pass1_steps,
        )
        return {
            "id": sid,
            "question": question,
            "answer": sample["answer"],
            "answer_aliases": sample.get("answer_aliases", []),
            "pass1_baseline": _summarise_agent(pass1_result),
            "pass2_augment": None,
            "pass3_enriched": None,
            "overhead": {"tokens": 0, "n_triplets_added": 0},
            "efficiency_gain": {"delta_tokens": 0, "delta_tool_calls": 0, "delta_steps": 0},
            "augmentation_skipped": True,
        }

    # --- Snapshot KG state before Pass 2 (first mutation) -------------------
    ent_count_before = _mcp_module._next_ent_id
    rel_count_before = _mcp_module._next_rel_id
    _mcp_module._edit_log.clear()

    logger.debug("[%s] Pass 2 augment (edit agent, no retrieval) …", sid)
    pass2_edits, pass2_token_stats = _augment_from_pass1(
        client=client,
        model_name=model_name,
        question=question,
        pass1_response=pass1_result["response"],
        temperature=temperature,
        max_steps=max_steps,
    )
    n_added = sum(1 for e in pass2_edits if e["op"] == "add")
    logger.debug(
        "[%s] Pass 2 done – %d triplets added, %d tokens",
        sid, n_added, pass2_token_stats["total_tokens"],
    )

    # --- Pass 3: enriched graph, read-only ----------------------------------
    logger.debug("[%s] Pass 3 enriched …", sid)
    pass3_result = agent_qa(
        client=client,
        model_name=model_name,
        question=question,
        tool_schemas=readonly_schemas,
        tool_dispatch=readonly_dispatch,
        system_prompt=_build_augment_enriched_system_prompt(),
        max_steps=max_steps,
        temperature=temperature,
        question_entities=question_entities,
    )
    logger.debug(
        "[%s] Pass 3 done – %d steps, %d tool calls, %d tokens",
        sid,
        len(pass3_result["logs"]),
        len(pass3_result["tool_calls"]),
        pass3_result["token_statistics"]["total_tokens"],
    )

    # --- Revert KG edits ----------------------------------------------------
    _revert_kg_edits(pass2_edits, ent_count_before, rel_count_before)
    logger.debug("[%s] KG reverted to pre-Pass-2 state", sid)

    p1 = _summarise_agent(pass1_result)
    p2 = {
        "kg_edits": pass2_edits,
        "n_kg_edits_added": n_added,
        "token_statistics": pass2_token_stats,
    }
    p3 = _summarise_agent(pass3_result)

    tok1 = p1["token_statistics"]["total_tokens"]
    tok3 = p3["token_statistics"]["total_tokens"]

    return {
        "id": sid,
        "question": question,
        "answer": sample["answer"],
        "answer_aliases": sample.get("answer_aliases", []),
        "pass1_baseline": p1,
        "pass2_augment": p2,
        "pass3_enriched": p3,
        # Overhead: cost of the single extraction call
        "overhead": {
            "tokens": pass2_token_stats["total_tokens"],
            "n_triplets_added": n_added,
        },
        # Efficiency gain: cost reduction on the enriched graph vs baseline
        "efficiency_gain": {
            "delta_tokens": tok3 - tok1,
            "delta_tool_calls": p3["n_tool_calls"] - p1["n_tool_calls"],
            "delta_steps": p3["n_steps"] - p1["n_steps"],
        },
    }


# ---------------------------------------------------------------------------
# Flat record helper for the evaluator
# ---------------------------------------------------------------------------
def _augment_result_to_eval_records(r: dict) -> tuple[dict, dict]:
    """Convert an augment result to two flat evaluator-compatible records.

    Pass 2 is a one-shot extraction call with no answer text, so only
    Pass 1 (baseline) and Pass 3 (enriched) are returned for evaluation.
    """
    base = {k: r[k] for k in ("id", "question", "answer", "answer_aliases")}

    def _flat(pass_dict: dict, suffix: str) -> dict:
        rec = dict(base)
        rec["response"] = pass_dict["response"]
        rec["id"] = f"{base['id']}__{suffix}"
        rec["n_steps"] = pass_dict["n_steps"]
        rec["n_tool_calls"] = pass_dict["n_tool_calls"]
        rec["token_statistics"] = pass_dict["token_statistics"]
        return rec

    p3 = r["pass3_enriched"]
    return (
        _flat(r["pass1_baseline"], "pass1_baseline"),
        _flat(p3, "pass3_enriched") if p3 is not None else None,
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------
@hydra.main(config_path="config", config_name="stage3_qa_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Output directory: %s", output_dir)

    # ---- Config ------------------------------------------------------------------
    mode = cfg.test.get("mode", "augment_qa")
    if mode not in ("batch_qa", "augment_qa", "report_only"):
        raise ValueError(f"test.mode must be 'batch_qa', 'augment_qa', or 'report_only', got '{mode}'")

    # ---- report_only: regenerate summary from an existing augment_results.jsonl ----
    if mode == "report_only":
        pass3_file = cfg.test.get("pass3_file", None)
        if not pass3_file:
            raise ValueError("test.pass3_file must be set when mode=report_only")
        _print_augment_summary(pass3_file, dataset_name=cfg.dataset.data_name)
        return

    n_threads = cfg.test.get("n_threads", 1)
    max_steps = cfg.test.get("max_steps", 10)
    batch_size = cfg.test.get("batch_size", 5)  # used in batch_qa mode only

    # augment_qa mutates shared KG state → must be single-threaded
    if mode == "augment_qa" and n_threads > 1:
        logger.warning("augment_qa mode requires single-threaded execution. Forcing n_threads=1.")
        n_threads = 1

    logger.info("Mode: %s | n_threads: %d", mode, n_threads)

    # ---- Load KG + docs ----------------------------------------------------------
    _init_retriever(cfg)

    # ---- OpenAI client -----------------------------------------------------------
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = cfg.llm.model_name_or_path
    temperature = 1.0 if model_name in ("gpt-5-mini", "gpt-5-nano") else 0.0

    # ---- Load test data ----------------------------------------------------------
    _, model_config = utils.load_model_from_pretrained(cfg.graph_retriever.model_path)
    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )
    test_data = qa_data.raw_test_data

    max_samples = cfg.test.get("max_test_samples", -1)
    if max_samples > 0:
        test_data = test_data[:max_samples]

    # ---- Resume support ----------------------------------------------------------
    processed_ids: set[str] = set()
    resume_path = cfg.test.get("resume", None)
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        try:
            with open(resume_path) as f:
                for line in f:
                    item = json.loads(line)
                    processed_ids.add(item["id"])
        except Exception as exc:
            logger.error("Could not resume: %s", exc)

    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    write_lock = threading.Lock()

    # =========================================================================
    # MODE 1 – batch_qa
    # =========================================================================
    if mode == "batch_qa":
        batch_system_prompt = _build_batch_system_prompt()

        # Group into batches, skipping already-processed IDs
        pending = [s for s in test_data if s["id"] not in processed_ids]
        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        logger.info(
            "batch_qa: %d samples → %d batches of up to %d",
            len(pending), len(batches), batch_size,
        )

        def _run_batch(batch: list[dict]) -> list[dict]:
            try:
                return run_batch_qa(
                    client=client,
                    model_name=model_name,
                    samples=batch,
                    max_steps=max_steps,
                    temperature=temperature,
                    system_prompt=batch_system_prompt,
                )
            except Exception as exc:
                logger.error("Batch failed (%s): %s", [s["id"] for s in batch], exc)
                return []

        with open(prediction_path, "w") as fout:
            if n_threads <= 1:
                for batch in tqdm(batches, desc="Batch QA"):
                    for rec in _run_batch(batch):
                        fout.write(json.dumps(rec) + "\n")
                    fout.flush()
            else:
                with ThreadPool(n_threads) as pool:
                    for results in tqdm(
                        pool.imap(_run_batch, batches),
                        total=len(batches),
                        desc="Batch QA",
                    ):
                        with write_lock:
                            for rec in results:
                                fout.write(json.dumps(rec) + "\n")
                            fout.flush()

    # =========================================================================
    # MODE 2 – augment_qa
    # =========================================================================
    else:
        # ---- Load Pass 1 cache (optional) --------------------------------------------
        pass1_cache: dict[str, dict] | None = None
        pass1_cache_file = cfg.test.get("pass1_cache_file", None)
        if pass1_cache_file:
            logger.info("Loading Pass 1 cache from %s", pass1_cache_file)
            pass1_cache = {}
            with open(pass1_cache_file) as f:
                for line in f:
                    rec = json.loads(line)
                    pass1_cache[rec["id"]] = rec
            logger.info("Loaded %d cached Pass 1 results", len(pass1_cache))

        pending = [s for s in test_data if s["id"] not in processed_ids]
        logger.info("augment_qa: %d samples (single-threaded)", len(pending))

        # Two output files: full augment results + flat eval-compatible records
        augment_path = os.path.join(output_dir, "augment_results.jsonl")
        eval_path = os.path.join(output_dir, "prediction.jsonl")  # for evaluator

        with open(augment_path, "w") as fout_aug, open(eval_path, "w") as fout_eval:
            for sample in tqdm(pending, desc="Augment QA"):
                try:
                    result = run_augment_qa(
                        client=client,
                        model_name=model_name,
                        sample=sample,
                        max_steps=max_steps,
                        temperature=temperature,
                        pass1_cache=pass1_cache,
                    )
                except Exception as exc:
                    logger.error("augment_qa failed on %s: %s", sample["id"], exc)
                    continue

                fout_aug.write(json.dumps(result) + "\n")
                fout_aug.flush()

                p1_rec, p3_rec = _augment_result_to_eval_records(result)
                fout_eval.write(json.dumps(p1_rec) + "\n")
                if p3_rec is not None:
                    fout_eval.write(json.dumps(p3_rec) + "\n")
                fout_eval.flush()

        # Print efficiency summary
        _print_augment_summary(augment_path, dataset_name=cfg.dataset.data_name)

    # ---- Evaluation (on flat prediction.jsonl) -----------------------------------
    evaluator = instantiate(cfg.qa_evaluator, prediction_file=prediction_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)


# ---------------------------------------------------------------------------
# Augment summary printer
# ---------------------------------------------------------------------------
def _print_augment_summary(augment_path: str, dataset_name: str = "hotpotqa") -> None:
    """Print aggregate statistics for an augment_qa run.

    Args:
        augment_path: Path to augment_results.jsonl.
        dataset_name: Dataset identifier used for inline per-question scoring
            (e.g. "hotpotqa_test", "musique_test", "2wikimultihopqa_test").
    """
    records: list[dict] = []
    try:
        with open(augment_path) as f:
            for line in f:
                records.append(json.loads(line))
    except Exception:
        return

    if not records:
        return

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    augmented = [r for r in records if r.get("pass3_enriched") is not None]
    skipped   = [r for r in records if r.get("pass3_enriched") is None]

    p1_steps  = [r["pass1_baseline"]["n_steps"] for r in augmented]
    p1_calls  = [r["pass1_baseline"]["n_tool_calls"] for r in augmented]
    p1_tokens = [r["pass1_baseline"]["token_statistics"]["total_tokens"] for r in augmented]

    p3_steps  = [r["pass3_enriched"]["n_steps"] for r in augmented]
    p3_calls  = [r["pass3_enriched"]["n_tool_calls"] for r in augmented]
    p3_tokens = [r["pass3_enriched"]["token_statistics"]["total_tokens"] for r in augmented]
    p2_tokens = [r["pass2_augment"]["token_statistics"]["total_tokens"] for r in augmented]
    n_edits   = [r["pass2_augment"]["n_kg_edits_added"] for r in augmented]
    gain_calls = [r["efficiency_gain"]["delta_tool_calls"] for r in augmented]

    def _outcome_counts(deltas: list[int]) -> tuple[int, int, int]:
        improved  = sum(1 for d in deltas if d < 0)
        unchanged = sum(1 for d in deltas if d == 0)
        regressed = sum(1 for d in deltas if d > 0)
        return improved, unchanged, regressed

    n = len(records)
    n_aug = len(augmented)
    gi, gu, gr = _outcome_counts(gain_calls)

    # --- Inline per-question accuracy (Part 4) --------------------------------
    from gfmrag.workflow._eval_utils import score_record  # noqa: PLC0415

    p1_em_list: list[float] = []
    p3_em_list: list[float] = []
    p1_f1_list: list[float] = []
    p3_f1_list: list[float] = []
    for r in augmented:
        answer = r.get("answer", "")
        aliases = r.get("answer_aliases", [])
        s1 = score_record(r["pass1_baseline"]["response"], answer, aliases, dataset_name)
        s3 = score_record(r["pass3_enriched"]["response"], answer, aliases, dataset_name)
        p1_em_list.append(s1["em"])
        p1_f1_list.append(s1["f1"])
        p3_em_list.append(s3["em"])
        p3_f1_list.append(s3["f1"])
        if s3["em"] >= s1["em"]:
            pass
            # logger.info(
            #     "EM improved: %s | Q: %s | A: %s \n P1: %s \n P3: %s \n\n",
            #     r["id"], r["question"], answer, r["pass1_baseline"]["response"], r["pass3_enriched"]["response"]
            # )
        else:
            logger.info(
                "EM not improved: %s | Q: %s | A: %s \n P1: %s \n P3: %s\n\n",
                r["id"], r["question"], answer, r["pass1_baseline"]["response"], r["pass3_enriched"]["response"]
            )

    p1_em = _mean(p1_em_list)
    p3_em = _mean(p3_em_list)
    p1_f1 = _mean(p1_f1_list)
    p3_f1 = _mean(p3_f1_list)
    triplet_utilized = sum(1 for e1, e3 in zip(p1_em_list, p3_em_list) if e3 > e1)
    # --------------------------------------------------------------------------

    w = 28
    print("\n" + "=" * 72)
    print("AUGMENTATION SUMMARY")
    print("=" * 72)
    print(f"  Samples evaluated              : {n}")
    print(f"  Samples augmented (Pass 2+3)   : {n_aug}  ({len(skipped)} skipped – Pass 1 ≤5 steps)")
    if augmented:
        print(f"  Avg KG triplets added (Pass 2) : {_mean(n_edits):.2f}")
        print(f"  Avg tokens for extraction      : {_mean(p2_tokens):.0f}")
    print()
    print(f"  {'Metric':<{w}} {'Pass1 (baseline)':>16} {'Pass3 (enriched)':>16}")
    print(f"  {'-'*w} {'-'*16} {'-'*16}")
    p3_steps_str  = f"{_mean(p3_steps):>16.2f}"  if augmented else f"{'N/A':>16}"
    p3_calls_str  = f"{_mean(p3_calls):>16.2f}"  if augmented else f"{'N/A':>16}"
    p3_tokens_str = f"{_mean(p3_tokens):>16.0f}"  if augmented else f"{'N/A':>16}"
    print(f"  {'Steps (mean)':<{w}} {_mean(p1_steps):>16.2f} {p3_steps_str}")
    print(f"  {'Tool calls (mean)':<{w}} {_mean(p1_calls):>16.2f} {p3_calls_str}")
    print(f"  {'Tokens (mean)':<{w}} {_mean(p1_tokens):>16.0f} {p3_tokens_str}")
    if augmented:
        print()
        print(f"  EFFICIENCY GAIN  (Pass3 vs Pass1 – value of enriched graph)")
        print(f"    Δ tool calls : {_mean(gain_calls):>+.2f}  "
              f"(improved {gi} / unchanged {gu} / regressed {gr})")
        print(f"    Δ tokens     : {_mean([r['efficiency_gain']['delta_tokens'] for r in augmented]):>+.0f}")
        print(f"    Δ steps      : {_mean([r['efficiency_gain']['delta_steps'] for r in augmented]):>+.2f}")
    print()
    print(f"  ACCURACY  (inline scoring, dataset={dataset_name})")
    print(f"    {'Metric':<20}  {'Pass1':>10}  {'Pass3':>10}  {'Delta':>10}")
    print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    p3_em_str = f"{p3_em:>10.3f}" if p3_em_list else f"{'N/A':>10}"
    p3_f1_str = f"{p3_f1:>10.3f}" if p3_f1_list else f"{'N/A':>10}"
    delta_em_str = f"{p3_em - p1_em:>+10.3f}" if p3_em_list else f"{'N/A':>10}"
    delta_f1_str = f"{p3_f1 - p1_f1:>+10.3f}" if p3_f1_list else f"{'N/A':>10}"
    print(f"    {'EM':<20}  {p1_em:>10.3f}  {p3_em_str}  {delta_em_str}")
    print(f"    {'F1':<20}  {p1_f1:>10.3f}  {p3_f1_str}  {delta_f1_str}")
    if augmented and p3_em_list:
        print()
        print(f"  TRIPLET UTILIZATION")
        print(f"    Questions where Pass3 EM > Pass1 EM : {triplet_utilized}/{n_aug}  ({triplet_utilized / n_aug:.1%})")
    print("=" * 72)


if __name__ == "__main__":
    main()
