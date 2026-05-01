import json
import logging
import os
import warnings
from multiprocessing.dummy import Pool as ThreadPool

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*",
)

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag import GFMRetriever
from gfmrag.evaluation import RetrievalEvaluator
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder
from gfmrag.ultra import query_utils

# A logger for this file
logger = logging.getLogger(__name__)

_TOKEN_STAT_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens")


def _merge_token_statistics(*stats: dict) -> dict:
    return {k: sum(s.get(k, 0) for s in stats if s) for k in _TOKEN_STAT_KEYS}


def agent_reasoning(
    cfg: DictConfig,
    gfmrag_retriever: GFMRetriever,
    llm: BaseLanguageModel,
    qa_prompt_builder: QAPromptBuilder,
    query: str,
) -> dict:
    step = 1
    current_query = query
    thoughts: list[str] = []
    retrieved_docs = gfmrag_retriever.retrieve(current_query, top_k=cfg.test.top_k)
    logs = []
    while step <= cfg.test.max_steps:
        message = qa_prompt_builder.build_input_prompt(
            current_query, retrieved_docs, thoughts
        )
        response, token_statistics = llm.generate_sentence(message)

        if isinstance(response, Exception):
            raise response from None

        thoughts.append(response)

        logs.append(
            {
                "step": step,
                "query": current_query,
                "retrieved_docs": retrieved_docs,
                "response": response,
                "thoughts": thoughts,
                "token_statistics": token_statistics,
            }
        )

        if "So the answer is:" in response:
            break

        step += 1

        new_ret_docs = gfmrag_retriever.retrieve(response, top_k=cfg.test.top_k)

        retrieved_docs_dict = {doc["title"]: doc for doc in retrieved_docs}
        for doc in new_ret_docs:
            if doc["title"] in retrieved_docs_dict:
                if doc["norm_score"] > retrieved_docs_dict[doc["title"]]["norm_score"]:
                    retrieved_docs_dict[doc["title"]]["score"] = doc["score"]
                    retrieved_docs_dict[doc["title"]]["norm_score"] = doc["norm_score"]
            else:
                retrieved_docs_dict[doc["title"]] = doc
        # Sort the retrieved docs by score
        retrieved_docs = sorted(
            retrieved_docs_dict.values(), key=lambda x: x["norm_score"], reverse=True
        )
        # Only keep the top k
        retrieved_docs = retrieved_docs[: cfg.test.top_k]

    final_response = " ".join(thoughts)
    merged_token_statistics = _merge_token_statistics(
        *(log["token_statistics"] for log in logs)
    )
    return {
        "response": final_response,
        "retrieved_docs": retrieved_docs,
        "logs": logs,
        "token_statistics": merged_token_statistics,
    }


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    gfmrag_retriever = GFMRetriever.from_config(cfg)
    llm = instantiate(cfg.llm)
    agent_prompt_builder = QAPromptBuilder(cfg.agent_prompt)
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    test_data = gfmrag_retriever.qa_data.raw_test_data
    max_samples = (
        cfg.test.max_test_samples if cfg.test.max_test_samples > 0 else len(test_data)
    )
    processed_data = {}
    if cfg.test.resume:
        logger.info(f"Resuming from previous prediction {cfg.test.resume}")
        try:
            with open(cfg.test.resume) as f:
                for line in f:
                    result = json.loads(line)
                    processed_data[result["id"]] = result
        except Exception as e:
            logger.error(f"Could not resume from previous prediction {e}")

    samples = [test_data[i] for i in range(max_samples)]

    def process_sample(sample: dict) -> dict | Exception:
        try:
            if sample["id"] in processed_data:
                return processed_data[sample["id"]]

            query = sample["question"]
            agent_result = agent_reasoning(
                cfg, gfmrag_retriever, llm, agent_prompt_builder, query
            )

            retrieved_docs = agent_result["retrieved_docs"]
            message = qa_prompt_builder.build_input_prompt(query, retrieved_docs)
            qa_response, token_statistics = llm.generate_sentence(message)

            if isinstance(qa_response, Exception):
                return qa_response

            merged_token_statistics = _merge_token_statistics(
                agent_result["token_statistics"], token_statistics
            )

            return {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_aliases": sample.get("answer_aliases", []),
                "supporting_facts": sample["supporting_facts"],
                "response": qa_response,
                "retrieved_docs": retrieved_docs,
                "logs": agent_result["logs"],
                "token_statistics": merged_token_statistics,
            }
        except Exception as e:
            return e

    max_workers = min(10, cfg.test.get("n_threads", 20))

    def write_result(f, result: dict | Exception) -> None:
        if isinstance(result, Exception):
            logger.error(f"Error processing sample: {result}")
            return
        f.write(json.dumps(result) + "\n")
        f.flush()

    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        if samples:
            logger.info("Running first sample sequentially to warm up retriever...")
            write_result(f, process_sample(samples[0]))

        if len(samples) > 1:
            with ThreadPool(max_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(process_sample, samples[1:]),
                    total=len(samples) - 1,
                ):
                    write_result(f, result)

    result_path = os.path.join(output_dir, "prediction.jsonl")
    # Evaluation
    evaluator = instantiate(cfg.qa_evaluator, prediction_file=result_path)
    metrics = evaluator.evaluate()
    query_utils.print_metrics(metrics, logger)

    # Eval retrieval results
    retrieval_evaluator = RetrievalEvaluator(prediction_file=result_path)
    retrieval_metrics = retrieval_evaluator.evaluate()
    query_utils.print_metrics(retrieval_metrics, logger)


if __name__ == "__main__":
    main()
