# AG-RAG: Rethinking Graph RAG Framework Design for Multi-hop Question Answering in the Agentic Era


## Dependencies

- Python 3.12
- CUDA 12 and above

## Installation

Conda provides an easy way to install the CUDA development toolkit which is required by GFM-RAG

Install packages
```bash
conda create -n gfmrag python=3.12
conda activate gfmrag
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1 # Replace with your desired CUDA version
pip install gfmrag
```

### Prepare Data

For data preparation and preprocessing, please refer to Stages 1 and 2 of [GFM-RAG](https://github.com/RManLuo/gfm-rag).


### Reproduce results

####
```
LLM="gpt-4o-mini" # or "gpt-5-mini"
DATA_NAME="hotpotqa" # or "musique" or "2wikimultihopqa"
```

#### GFM-RAG Baseline

```
python -m gfmrag.workflow.stage3_qa_inference \
llm.model_name_or_path=${LLM} \
qa_prompt=${DATA_NAME} \
qa_evaluator=${DATA_NAME} \
dataset.data_name=${DATA_NAME}_test
```

#### GFM-RAG + IRCoT Baseline
```
python -m gfmrag.workflow.stage3_qa_ircot_inference \
llm.model_name_or_path=${LLM} \
qa_prompt=${DATA_NAME} \
qa_evaluator=${DATA_NAME} \
dataset.data_name=${DATA_NAME}_test
```

#### AG-RAG
```
python -m gfmrag.workflow.stage3_qa_inference_mcp \ 
llm.model_name_or_path=${LLM} \
qa_prompt=${DATA_NAME} \
qa_evaluator=${DATA_NAME} \
dataset.data_name=${DATA_NAME}_test
```

#### AG-RAG augment QA
```
python -m gfmrag.workflow.stage3_qa_inference_mcp_augment \
test.mode=augment_qa \
llm.model_name_or_path=${LLM} \
qa_prompt=${DATA_NAME} \
qa_evaluator=${DATA_NAME} \
dataset.data_name=${DATA_NAME}_test
```

#### AG-RAG batch QA
```
python -m gfmrag.workflow.stage3_qa_inference_mcp_augment \
test.mode=batch_qa \
llm.model_name_or_path=${LLM} \
qa_prompt=${DATA_NAME} \
qa_evaluator=${DATA_NAME} \
dataset.data_name=${DATA_NAME}_test
```

## Acknowledgements

This work builds heavily on the contributions of [GFM-RAG](https://github.com/RManLuo/gfm-rag)

Commit: 56149d76eadafeeaf224f9fe2bb67e42a239c7bd
