import json
import logging
import os
import threading
import time
from multiprocessing.dummy import Pool as ThreadPool

import dotenv
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from tqdm import tqdm

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.ultra import query_utils

# Import the MCP tool functions and initialiser directly
from gfmrag.workflow.mcp_server import (
    _init_retriever,
    add_kg_triplet,
    delete_kg_triplet,
    get_batch_neighbor_entities,
    get_kg_edit_log,
    get_neighbor_entities,
    retrieve_documents,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------
_BASE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_neighbor_entities",
            "description": (
                "Retrieve the neighbor entities of a given entity in the "
                "knowledge graph. Returns neighbours grouped by relation type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "The entity name to look up (case-insensitive).",
                    },
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_batch_neighbor_entities",
            "description": (
                "Retrieve neighbor entities for multiple entities in a single "
                "call. More efficient than calling get_neighbor_entities "
                "repeatedly when several entities are already known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entity names to look up (case-insensitive).",
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Optional relation type filter applied to all entities.",
                    },
                },
                "required": ["entity_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Retrieve documents associated with one or more entities in the "
                "knowledge graph. When multiple entities are provided their "
                "document scores are aggregated so that documents relevant to "
                "all of them are ranked highest. Each document has a title, "
                "content text, and relevance score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of entity names to look up (case-insensitive).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (default 3).",
                        "default": 3,
                    },
                },
                "required": ["entity_names"],
            },
        },
    },
]

_KG_EDIT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_kg_triplet",
            "description": (
                "Add a new (head, relation, tail) fact to the knowledge graph. "
                "Creates new entities or relations if they do not exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "head": {"type": "string", "description": "Source entity name."},
                    "relation": {"type": "string", "description": "Relation / edge label."},
                    "tail": {"type": "string", "description": "Target entity name."},
                },
                "required": ["head", "relation", "tail"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_kg_triplet",
            "description": (
                "Remove an existing (head, relation, tail) fact from the "
                "knowledge graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "head": {"type": "string", "description": "Source entity name."},
                    "relation": {"type": "string", "description": "Relation / edge label."},
                    "tail": {"type": "string", "description": "Target entity name."},
                },
                "required": ["head", "relation", "tail"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_kg_edit_log",
            "description": "Return the log of all KG edits made in this session.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Base tool dispatch (always available)
_BASE_TOOL_DISPATCH: dict[str, object] = {
    "get_neighbor_entities": get_neighbor_entities,
    "get_batch_neighbor_entities": get_batch_neighbor_entities,
    "retrieve_documents": retrieve_documents,
}

_KG_EDIT_TOOL_DISPATCH: dict[str, object] = {
    "add_kg_triplet": add_kg_triplet,
    "delete_kg_triplet": delete_kg_triplet,
    "get_kg_edit_log": get_kg_edit_log,
}

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_BASE = """\
You are an expert question-answering agent with access to a knowledge graph \
and a document corpus via tools:

1. **get_neighbor_entities(entity_name)** – explore the \
knowledge graph by listing an entity's neighbours.

2. **get_batch_neighbor_entities(entity_names)** – retrieve neighbours for \
multiple entities in one call; use this when you already know several entities \
to explore simultaneously.

3. **retrieve_documents(entity_names, top_k?)** – retrieve the most relevant \
documents associated with one or more entities. When multiple entities are \
provided, document scores are aggregated so documents relevant to all of \
them rank highest.

## Core assumptions
1. The answer to every user question can be found in the provided dataset
  (knowledge graph and/or document corpus). Do not speculate beyond the dataset.
  Do not use any external knowledge or your own general knowledge.

## Tool-use rules (strict)
1. Prefer the knowledge graph first:
   - Always attempt to answer using get_neighbor_entities before retrieving documents.
   - Only call retrieve_documents if you cannot obtain sufficient evidence from
     the knowledge graph neighbor exploration (including multi-hop exploration).

2. Batch tool calls:
   - If you anticipate needing multiple tool calls, plan them and execute them
     in a single tool round whenever possible (e.g., use get_batch_neighbor_entities
     to query multiple entities together), rather than spreading calls across
     multiple rounds.

3. Retrieval gating:
   - Retrieve documents only when the needed information cannot be found via
     knowledge graph neighbor traversal (including checking alternative entity names,
     aliases if available, and exploring relevant neighbors).

Your goal is to answer the user's question accurately. Follow this strategy:
- Identify entities mentioned in the question.
- Verify entity existence in the knowledge graph using get_neighbor_entities or
  get_batch_neighbor_entities.
- Traverse neighbors (and multi-hop paths) to gather evidence needed to answer.
  The reasoning process must be grounded in the retrieved entities and their relationships.
  Avoid making assumptions or fabricating connections that are not supported by the graph structure.
- If (and only if) the knowledge graph traversal is insufficient, call
  retrieve_documents with the smallest necessary set of entities and a small top_k.
- When you have enough evidence, provide a final answer on a new line in exactly this format:
  Evidence: <the reasoning path according to the retrieved information> Answer: <your concise answer with only a short term or phrase as the final answer>
- Be concise. Avoid repeating tool outputs verbatim.
- Do not use possible answers as search queries unless they are supported by evidence from previous queries.
  This is strictly prohibited, as it may compromise the integrity of the evaluation.

"""

_KG_EDIT_ADDENDUM = """\
You also have access to knowledge graph editing tools:

4. **add_kg_triplet(head, relation, tail)** – add a new fact to the knowledge graph.
   Use this to record inferred facts or new information discovered during reasoning.

5. **delete_kg_triplet(head, relation, tail)** – remove an incorrect or outdated fact.

6. **get_kg_edit_log()** – review all edits made in this session.

Use editing tools judiciously: only add facts that are clearly supported by evidence
already retrieved, and only delete facts that are demonstrably incorrect.

"""


def _build_system_prompt(enable_kg_edit: bool) -> str:
    if enable_kg_edit:
        return _SYSTEM_PROMPT_BASE + _KG_EDIT_ADDENDUM
    return _SYSTEM_PROMPT_BASE


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def _execute_tool_call(tool_call, tool_dispatch: dict) -> str:
    """Run a single tool call and return its string result."""
    fn_name = tool_call.function.name
    fn_args = json.loads(tool_call.function.arguments)
    fn = tool_dispatch.get(fn_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{fn_name}'"})
    try:
        return fn(**fn_args)
    except Exception as exc:
        print(fn_args)
        return json.dumps({"error": str(exc)})


def agent_qa(
    client: OpenAI,
    model_name: str,
    question: str,
    tool_schemas: list[dict],
    tool_dispatch: dict,
    system_prompt: str,
    max_steps: int = 10,
    retry: int = 3,
    temperature: float = 0.0,
    question_entities: list[str] | None = None,
    reasoning_effort: str = "low",
) -> dict:
    """Run the multi-turn MCP agent loop for a single question.

    Returns a dict with keys: response, logs, tool_calls, token_statistics.
    """

    # Seed the conversation with the system prompt and the user question
    user_content = f"Question: {question}"
    if question_entities:
        user_content += (
            "\n\nHint – the following entities from the question may exist "
            "in the knowledge graph (they might be useful starting points): "
            + ", ".join(question_entities)
        )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    all_tool_calls: list[dict] = []
    logs: list[dict] = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "input_cached_tokens": 0}

    for step in range(1, max_steps + 1):
        # Per-step timing accumulators (reset each step)
        api_call_ms: float = 0.0
        tool_exec_total_ms: float = 0.0

        # Call the LLM with tools
        error = None
        for attempt in range(retry):
            try:
                t1 = time.time()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    temperature=1.0,
                    top_p=0.95,
                )
                api_call_ms = (time.time() - t1) * 1000
                # print(f"Time taken for API call: {api_call_ms:.0f} ms")
                break
            except Exception as exc:
                logger.error("LLM call failed (attempt %d): %s", attempt + 1, exc)
                error = exc
                time.sleep(5)
        else:
            # All retries exhausted
            raise error  # type: ignore[misc]

        # Accumulate token usage
        if response.usage:
            total_tokens["input_tokens"] += response.usage.prompt_tokens
            total_tokens["output_tokens"] += response.usage.completion_tokens
            total_tokens["total_tokens"] += response.usage.total_tokens
            total_tokens["input_cached_tokens"] += response.usage.prompt_tokens_details.cached_tokens

        choice = response.choices[0]
        assistant_msg = choice.message

        # Append the raw assistant message to the conversation
        messages.append(assistant_msg.model_dump())

        step_log: dict = {"step": step, "role": "assistant"}

        # If the model produced text content (possibly alongside tool calls)
        if assistant_msg.content:
            step_log["content"] = assistant_msg.content

        # Handle tool calls
        if assistant_msg.tool_calls:
            tool_results = []
            for tc in assistant_msg.tool_calls:
                t_tool = time.time()
                result_str = _execute_tool_call(tc, tool_dispatch)
                tool_exec_total_ms += (time.time() - t_tool) * 1000
                tool_results.append(
                    {
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                        "result": result_str,
                    }
                )
                # Feed the tool result back into the conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )
            step_log["tool_calls"] = tool_results
            step_log["api_call_ms"] = round(api_call_ms, 1)
            step_log["tool_exec_ms"] = round(tool_exec_total_ms, 1)
            all_tool_calls.extend(tool_results)
        else:
            # No tool calls – the model has finished reasoning
            step_log["finish_reason"] = str(choice.finish_reason)
            step_log["api_call_ms"] = round(api_call_ms, 1)
            step_log["tool_exec_ms"] = 0.0
            logs.append(step_log)
            break

        logs.append(step_log)

    # Extract final response text
    final_text = assistant_msg.content or ""

    return {
        "response": final_text.strip(),
        "logs": logs,
        "tool_calls": all_tool_calls,
        "token_statistics": total_tokens,
    }


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------
@hydra.main(config_path="config", config_name="stage3_qa_inference", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info("Config:\n %s", OmegaConf.to_yaml(cfg))
    logger.info("Current working directory: %s", os.getcwd())
    logger.info("Output directory: %s", output_dir)

    # ---- Config flags --------------------------------------------------------
    n_threads = cfg.test.get("n_threads", 1)
    enable_kg_edit = cfg.test.get("enable_kg_edit", False)
    max_steps = cfg.test.get("max_steps", 10)

    if enable_kg_edit and n_threads > 1:
        logger.warning(
            "enable_kg_edit=True requires single-threaded execution. "
            "Forcing n_threads=1."
        )
        n_threads = 1

    # ---- Initialise the MCP server's data (KG + docs) in-process -----------
    _init_retriever(cfg)

    # ---- Build tool schemas and dispatch based on config -------------------
    tool_schemas = list(_BASE_TOOL_SCHEMAS)
    tool_dispatch = dict(_BASE_TOOL_DISPATCH)
    if enable_kg_edit:
        tool_schemas.extend(_KG_EDIT_TOOL_SCHEMAS)
        tool_dispatch.update(_KG_EDIT_TOOL_DISPATCH)
        logger.info("KG editing tools enabled.")

    system_prompt = _build_system_prompt(enable_kg_edit)

    # ---- Prepare the OpenAI client -----------------------------------------
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key="",
    # )
    model_name = cfg.llm.model_name_or_path
    temperature = 1.0

    # ---- Load test data -----------------------------------------------------
    _, model_config = utils.load_model_from_pretrained(cfg.graph_retriever.model_path)
    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
    )
    test_data = qa_data.raw_test_data

    max_samples = cfg.test.get("max_test_samples", -1)
    if max_samples <= 0:
        max_samples = len(test_data)
    test_data = test_data[:max_samples]

    # ---- Resume support -----------------------------------------------------
    processed_ids: dict[str, dict] = {}
    resume_path = cfg.test.get("resume", None)
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        try:
            with open(resume_path) as f:
                for line in f:
                    item = json.loads(line)
                    processed_ids[item["id"]] = item
        except Exception as exc:
            logger.error("Could not resume: %s", exc)

    # ---- Worker function (called by each thread) ----------------------------
    def _run_sample(sample: dict) -> dict | None:
        sid = sample["id"]

        if sid in processed_ids:
            return processed_ids[sid]

        question = sample["question"]
        question_entities = sample.get("question_entities", [])

        try:
            agent_result = agent_qa(
                client=client,
                model_name=model_name,
                question=question,
                tool_schemas=tool_schemas,
                tool_dispatch=tool_dispatch,
                system_prompt=system_prompt,
                max_steps=max_steps,
                question_entities=question_entities,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("Agent failed on sample %s: %s", sid, exc)
            return None

        return {
            "id": sid,
            "question": question,
            "answer": sample["answer"],
            "answer_aliases": sample.get("answer_aliases", []),
            "response": agent_result["response"],
            "logs": agent_result["logs"],
            "tool_calls": agent_result["tool_calls"],
            "token_statistics": agent_result["token_statistics"],
        }

    # ---- Run agent on each sample -------------------------------------------
    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    write_lock = threading.Lock()

    logger.info(
        "Running MCP Agent QA on %d samples with n_threads=%d", len(test_data), n_threads
    )

    with open(prediction_path, "w") as fout:
        if n_threads <= 1:
            # Single-threaded path (required when enable_kg_edit=True)
            for sample in tqdm(test_data, desc="MCP Agent QA"):
                result = _run_sample(sample)
                if result:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
        else:
            # Multi-threaded path
            with ThreadPool(n_threads) as pool:
                for result in tqdm(
                    pool.imap(_run_sample, test_data),
                    total=len(test_data),
                    desc="MCP Agent QA",
                ):
                    if result:
                        with write_lock:
                            fout.write(json.dumps(result) + "\n")
                            fout.flush()

    # ---- Evaluation ---------------------------------------------------------
    evaluator = instantiate(cfg.qa_evaluator, prediction_file=prediction_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)


if __name__ == "__main__":
    main()
