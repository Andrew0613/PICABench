<h1 align="center">
  <img src="https://picabench.github.io/static/img/icons/pica_logo.png" width="48" style="vertical-align: middle; margin-right: 8px;" />
  PICABench: How Far Are We from Physically Realistic Image Editing?
</h1>

<div align="center">

<i>Benchmark, evaluator, and data suite for physically realistic image editing.</i>

[![Website](https://img.shields.io/badge/Project-Website-007ec6?style=for-the-badge)](https://picabench.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2510.17681-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2510.17681)
[![Dataset](https://img.shields.io/badge/HF-Dataset%20(PICABench)-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Andrew613/PICABench)
[![Dataset](https://img.shields.io/badge/HF-Dataset%20(PICA--100K)-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Andrew613/PICA-100K)

</div>

---

<div align="center">
  <img src="https://picabench.github.io/static/img/picabench_teaser.png" alt="PICABench teaser" width="40%" />
</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Evaluation Pipelines](#evaluation-pipelines)
- [PICA-100K Training Data](#pica-100k-training-data)
- [Leaderboard & Qualitative Explorer](#leaderboard--qualitative-explorer)
- [Leaderboard Submission](#leaderboard-submission)
- [Citation](#citation)

## Overview

PICABench probes how far current editing models are from physically realistic image manipulation. It ties together:

- **PICABench benchmark** – physics-aware editing cases spanning eight laws across *Optics*, *Mechanics*, and *State Transition*, each labeled with superficial/intermediate/explicit difficulty tiers.
- **PICAEval metric** – region-grounded, QA-based verification with human-annotated regions of interest (ROIs) and spatially anchored yes/no questions.
- **PICA-100K dataset** – synthetic, video-derived training data that boosts physics consistency when used for fine-tuning.

The leaderboard shows that even top proprietary systems only reach ~60% accuracy, indicating a significant physics-awareness gap.

## ⚡ Quick Start

Evaluate your model's physics-aware editing in 3 steps:

```bash
# 1. Download benchmark data
huggingface-cli download Andrew613/PICABench \
  --repo-type dataset \
  --local-dir PICABench_data

# 2. Install dependencies (choose GPT or Qwen)
pip install openai Pillow tqdm huggingface_hub  # GPT-5
# or
pip install vllm transformers Pillow tqdm       # Qwen/vLLM

# 3. Run evaluation
export OPENAI_API_KEY="sk-..."
python PicaEval_gpt.py \
  --input_json_path PICABench_data/meta_info.json \
  --image_base_dir PICABench_data \
  --gpt_model gpt-5
```

Results will be saved as `meta_info_gpt_output_1024_crop_box_and_resize.json` with per-question accuracy and physics law breakdown.

## Installation

We recommend using a Python 3.10+ virtual environment:

```bash
conda create -n picabench python=3.10
conda activate picabench
```

Install dependencies based on your evaluation needs:

```bash
# For GPT-5 evaluation
pip install openai Pillow tqdm huggingface_hub

# For Qwen evaluation (with vLLM acceleration)
pip install vllm transformers
```

## Data Preparation

PICABench expects per-scene metadata in `meta_info.json` plus accompanying images under a shared base directory. Each item should include:

```jsonc
{
  "index": 1174,
  "input_path": "input_img/1174.jpg",
  "output_path": "output_img/1174.jpg",
  "edit_instruction": "Remove the tulip from the white vase and simultaneously eliminate every instance of it in the window reflection while keeping lighting and shading consistent.",
  "physics_category": "Optics",
  "physics_law": "Reflection",
  "edit_operation": "remove",
  "difficulty": "superficial",
  "annotated_qa_pairs": [
    {
      "question": "Is a tulip visible in the window reflection?",
      "answer": "No",
      "box": { "x": 101.25, "y": 476.90, "width": 169.44, "height": 202.96 }
    },
    {
      "question": "Does the interior of the white vase contain exactly zero tulips?",
      "answer": "Yes",
      "box": { "x": 327.96, "y": 485.99, "width": 209.80, "height": 206.21 }
    },
    {
      "question": "Is the vase's reflection aligned with the vase?",
      "answer": "Yes",
      "box": { "x": 117.24, "y": 496.29, "width": 363.74, "height": 183.41 }
    }
  ],
  "edit_area": [
    {
      "x": 117.24,
      "y": 496.29,
      "width": 363.74,
      "height": 183.41,
      "id": "BxnMC34B",
      "order": 1
    }
  ]
}
```

<details>
<summary><b>📋 Field Descriptions (click to expand)</b></summary>

- **`annotated_qa_pairs`**: List of QA dictionaries for physics verification. Each contains:
  - `question`: Yes/no question about physical correctness
  - `answer`: Ground truth ("Yes" or "No")
  - `box`: Region of interest `{x, y, width, height}` in 1024px canvas coordinates
  
- **`edit_area`**: Bounding boxes of edited regions (used for visualization cropping). Set to `"unknown"` if unavailable.

- **Visualization**: Scripts auto-generate cropped/annotated images in `visualization_annotated_qa_crop_box_and_resize/` under the base directory.

</details>

## Evaluation Pipelines

### 1. Qwen / vLLM (PICAEval)

```bash
python PicaEval_qwen.py \
  --input_json_path /path/to/meta_info.json \
  --image_base_dir /path/to/images \
  --model_path pretrained/Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --qa_field annotated_qa_pairs \
  --viz_mode crop_box_and_resize \
  --max_new_tokens 256 \
  --img_size 1024
```

Outputs:

- `<meta>_vllm_output_<img_size>[_mode].json` – per-QA predictions with `model_answer`, `model_response`, `model_explanation`, `is_correct`, and optional `visualization_path`.
- `<meta>_vllm_analysis_<img_size>[_mode].json` – aggregated accuracy by physics category, law, and operation.

### 2. GPT-based Evaluation (PICAEval)

```bash
export OPENAI_API_KEY="sk-..."
python PicaEval_gpt.py \
  --input_json_path /path/to/meta_info.json \
  --image_base_dir /path/to/images \
  --qa_field annotated_qa_pairs \
  --viz_mode crop_box_and_resize \
  --gpt_model gpt-5 \
  --max_attempts 5 \
  --reasoning_effort low
```
Outputs:
- `<meta>_gpt_output_<img_size>[_{mode}].json` – detailed results
- `<meta>_gpt_analysis_<img_size>[_{mode}].json` – accuracy statistics

Notes:

- Reuses the same JSON schema for inputs/outputs as the Qwen pipeline, enabling direct comparison.
- Images are base64-encoded and sent as data URLs; be mindful of API quotas and rate limits.

### 3. Non-edited Region Quality

```bash
python PicaEval_consistency.py \
  --input_json_path /path/to/meta_info.json \
  --image_base_dir /path/to/images \
  --qa_field annotated_qa_pairs \      # QA field name
  --viz_mode crop_box_and_resize       # visualization mode
```

Produces `<meta>_nonedited_metrics_output.json` and `_analysis.json`, containing masked PSNR/SSIM/LPIPS scores or whole-image fallbacks when edit regions are unavailable.

## PICA-100K Training Data

**Dataset**: [Andrew613/PICA-100K](https://huggingface.co/datasets/Andrew613/PICA-100K)

100K synthetic editing pairs derived from video frames, designed to improve physical realism in image editing models.

### Download

```bash
huggingface-cli download Andrew613/PICA-100K \
  --repo-type dataset \
  --local-dir data/PICA-100K
```

## Leaderboard & Qualitative Explorer

- Official leaderboard and gallery: [https://picabench.github.io](https://picabench.github.io)
- Eight physics laws × three difficulty tiers provide direct qualitative comparisons.
- PICAEval scores correlate strongly with human judgments (Elo study on the site).

## Leaderboard Submission

To submit your model's results to the PICABench leaderboard:

**Required Metrics:**
  - Accuracy (%) for each sub-category (Light Propagation, Reflection, Refraction, Light Source Effects, Deformation, Causality, Local State Transition, Global State Transition)
  - Overall Accuracy (%)

**Submission:**
Email your `*_analysis*.json` and `*_output*.json` files and model details to:
- [puyuandong01061313@gmail.com](mailto:puyuandong01061313@gmail.com)
## Citation

```bibtex
@article{pu2025picabench,
  title        = {PICABench: How Far Are We From Physically Realistic Image Editing?},
  author       = {Pu, Yuandong and Zhuo, Le and Han, Songhao and Xing, Jinbo and Zhu, Kaiwen and Cao, Shuo and Fu, Bin and Liu, Si and Li, Hongsheng and Qiao, Yu and Zhang, Wenlong and Chen, Xi and Liu, Yihao},
  journal      = {arXiv preprint arXiv:2510.17681},
  year         = {2025}
}
```

## License

This project is released under the Apache License 2.0.
