<div align="center">
    <img src="logo/air.png" alt="AIR Logo" width="100"/>

# AIR: Complex Instruction Generation via Automatic Iterative Refinement
</div>

## 🌟 Overview

AIR is a novel framework for generating complex instructions with constraints, significantly enhancing Large Language Models' ability to follow complex instructions. Our approach uses an innovative two-stage process:

1. **Initial Instruction Generation**: Generate base instructions from documents
2. **Iterative Refinement**: Enhance instructions through LLM-as-judge guidance

The framework produces more challenging and realistic instructions, leading to improved model performance on complex tasks.

## 🚀 Key Features

- **Automatic Iterative Refinement**: Novel approach to generate complex instructions
- **Constraint-aware Generation**: Instructions that better reflect real-world scenarios
- **Large-scale Dataset**: AIR-10K dataset with 10,000 complex instructions
- **Enhanced Performance**: Significant improvements over existing instruction-following methods

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Download Dolma Dataset
```bash
# Download dataset chunks
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data/dolma --include "*000_00000.parquet*"
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data/dolma --include "*001_00000.parquet*"
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data/dolma --include "*002_00000.parquet*"
```

### Initial Processing
```bash
# Convert data format
python ./init_process/data_acquire.py \
    --input_path ./data/dolma \
    --output_path ./data/dolma.jsonl

# Generate embeddings
python ./init_process/embeds_gene.py \
    --input_path ./data/dolma.jsonl \
    --output_path ./data/doc_embeds.jsonl

# Select diverse documents
python ./init_process/select_diverse_based_doc_embeds.py \
    --input_path ./data/dolma.jsonl \
    --embedding_path ./data/doc_embeds.jsonl \
    --output_path ./data/dolma_60k.jsonl

# Generate initial instructions
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./init_process/instruct_generate.py \
    -i ./data/dolma_60k.jsonl \
    -o ./data/dolma_init_process.jsonl \
    -m /path/llama3_70b_instruct

# Filter and score instructions
python ./init_process/instruct_score_filter.py \
    --input_path ./data/dolma_init_process.jsonl \
    --output_path ./data/dolma_init_process.jsonl
```

### Generate and Process Judge Data

#### scripts
```bash
# Generate judge data (max 5 iterations)
bash ./judge_data_gene/run_main.sh

# Process for SFT training
bash ./judge_data_process/data_process.sh
```

#### Models Used
- [Llama-3-8B-UltraChat](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT)
- Qwen-2.5-7B-UltraChat (Custom fine-tuned version) 
- [Llama-3-8B-Tulu](https://huggingface.co/Magpie-Align/Llama-3-8B-Tulu-330K)

#### Guidance Models Used
- [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) (for Llama series)
- [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) (for Qwen series)



## 🔄 Training

We support training using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) with the following models:

- [Llama-3-8B-UltraChat](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT)
- Qwen-2.5-7B-UltraChat (Custom fine-tuned version)
- [Llama-3-8B-Tulu](https://huggingface.co/Magpie-Align/Llama-3-8B-Tulu-330K)

<br><br>
