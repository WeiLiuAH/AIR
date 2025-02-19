# AIR: Complex Instruction Generation via Automatic Iterative Refinement
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Datasets-blue)](https://huggingface.co/datasets/MengGaoang/AIR)
[![arXiv](https://img.shields.io/badge/arXiv-2502.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2502.XXXXX)

This is the official implementation of our paper "AIR: Complex Instruction Generation via Automatic Iterative Refinement". We propose a novel Automatic Iterative Refinement (AIR) framework to generate complex instructions with constraints, significantly enhancing LLMs' ability to follow complex instructions.

## Installation

```bash
git clone https://github.com/LiuWeiHITees/AIR
cd AIR
pip install -r requirements.txt
```

## Dataset Preparation

### Download Dolma Dataset
```bash
# Download dataset chunks
huggingface-cli download --repo-type dataset  --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data/dolma --include "*000_00000.parquet*"
huggingface-cli download --repo-type dataset  --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data/dolma --include "*001_00000.parquet*"
huggingface-cli download --repo-type dataset  --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data/dolma --include "*002_00000.parquet*"
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

### Judge Data Generation
```bash
bash ./judge_data_gene/run_main.sh
```

### Judge Data Processing
```bash
bash ./judge_data_process/data_process_0207.sh
```

## Training

You can directly download our processed datasets from Huggingface:
- [Dataset Link](https://huggingface.co/datasets/[your-repo-name])

We support training using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). The following models are compatible with our framework:

- [Llama-3-8B-Tulu-330K](https://huggingface.co/Magpie-Align/Llama-3-8B-Tulu-330K)
- [Llama-3-Base-8B-SFT](https://huggingface.co/princeton-nlp/Llama-3-Base-8B-SFT)
- Qwen2.5-7B-UltraChat (Our fine-tuned version)

## Citation

If you find this work helpful, please cite our paper:
```bibtex
@article{air2025,
  title={},
  author={},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

## Acknowledgements

- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory)
- [Dolma Dataset](https://huggingface.co/datasets/emozilla/dolma-v1_7-cc_en_head)
