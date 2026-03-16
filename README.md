# QLoRA LLM Fine-Tuning — Llama 3.2 3B on MedQA

> Full fine-tuning pipeline with LoRA rank ablation study, quantization benchmarking, and production deployment on HuggingFace Spaces.

[![Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/aakthepaak/clinical-llm-demo)
[![Model](https://img.shields.io/badge/🤗%20Model-aakthepaak/clinical--llm--r32-orange)](https://huggingface.co/aakthepaak/clinical-llm-r32)
[![WandB](https://img.shields.io/badge/WandB-Experiment%20Tracking-yellow)](https://wandb.ai/aakarsh3110-stevens-institute-of-technology/clinical-llm)

---

## Overview

This project fine-tunes **Llama 3.2 3B Instruct** on the **MedQA** dataset (USMLE-style multiple choice questions) using **QLoRA** — 4-bit quantized low-rank adaptation. The core deliverable is a systematic **LoRA rank ablation study** (r=4, 8, 16, 32) evaluating the accuracy vs. parameter efficiency tradeoff, followed by quantization benchmarking and a fully deployed Gradio demo.

**Key result:** Fine-tuned r=32 achieves **59.2% accuracy vs. 10% baseline** — a 6x improvement — on 2,500 training samples.

---

## Architecture

```
medalpaca/medical_meadow_medqa
        │
        ▼
  Data Formatting
  (instruction + question → chat template)
        │
        ▼
  Llama 3.2 3B Instruct (4-bit, nf4)
  + QLoRA Adapters (r = 4 / 8 / 16 / 32)
        │
        ▼
  SFTTrainer (Unsloth + TRL)
  3 epochs | lr=2e-4 | batch=16 (effective)
        │
        ▼
  Evaluation (Accuracy + ROUGE-L)
        │
        ▼
  Merge r=32 → fp16
        │
        ├──► HuggingFace Hub (aakthepaak/clinical-llm-r32)
        │
        ├──► Quantization Benchmark (4-bit vs fp16)
        │
        ├──► Gradio Demo (HuggingFace Spaces)
        │
        └──► FastAPI + Docker (local serving)
```

---

## LoRA Rank Ablation Study

All runs trained on identical hyperparameters, varying only `r` and `lora_alpha` (convention: `lora_alpha = 2 × r`). Tracked on WandB.

| Rank | lora_alpha | Trainable Params | Train Loss | Accuracy | ROUGE-L |
|------|------------|-----------------|------------|----------|---------|
| Base (no fine-tune) | — | — | — | 0.100 | 0.128 |
| r=4 | 8 | 6,078,464 (0.19%) | 1.1689 | 0.560 | 0.319 |
| r=8 | 16 | 12,156,928 (0.38%) | 1.1389 | 0.576 | 0.356 |
| r=16 | 32 | 24,313,856 (0.75%) | 1.1013 | 0.560 | 0.342 |
| r=32 | 64 | ~48M (1.50%) | 1.0496 | **0.592** | **0.355** |

**Finding:** Diminishing returns after r=8 for ROUGE-L, but r=32 gives the best accuracy. The base model's low score is a format compliance issue — it generates verbose explanations rather than the concise single-letter answer the benchmark expects. Fine-tuning teaches format adherence as much as domain knowledge.

---

## Quantization Benchmark

Merged r=32 model benchmarked across quantization levels on L4 GPU (20 inference samples each).

| Quantization | Avg Latency | Accuracy |
|-------------|-------------|----------|
| 4-bit (nf4) | 0.458s | 0.588 |
| fp16 | 0.583s | 0.596 |

**Finding:** 4-bit is **27% faster** with negligible accuracy loss (0.8%) — strong production argument for quantization. 8-bit skipped due to confirmed Unsloth bug [#2679](https://github.com/unslothai/unsloth/issues/2679).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Dataset | medalpaca/medical_meadow_medqa |
| Training samples | 2,250 (train) / 250 (val) |
| Quantization (training) | 4-bit nf4 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| lora_dropout | 0 |
| Epochs | 3 |
| Batch size | 2 (effective 16 with grad accum) |
| Learning rate | 2e-4 |
| Warmup steps | 50 |
| Max seq length | 2048 |
| Hardware | Google Colab L4 (23.7GB VRAM) |
| Framework | Unsloth 2026.3.4 + Transformers 5.2.0 |

---

## Project Structure

```
qlora-llm-finetuning/
├── Fine_Tuning_Project.ipynb   ← Full training + eval + benchmarking notebook
└── README.md
```

**Related repos:**
- [`clinical-llm-api`](https://github.com/aakarsh31/clinical-llm-api) — FastAPI + Docker inference service

---

## Quickstart

### Run inference from HuggingFace Hub

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="aakthepaak/clinical-llm-r32",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = """<|system|> You are a clinical medical assistant, answer the following question.
<|user|> Please answer with one of the option in the bracket.
Q: A 23-year-old presents with burning urination. Which antibiotic is safe in pregnancy?
{'A': 'Ciprofloxacin', 'B': 'Nitrofurantoin', 'C': 'Tetracycline', 'D': 'Doxycycline'}
<|assistant|>"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

### Run locally with Docker (requires NVIDIA GPU)

```bash
git clone https://github.com/aakarsh31/clinical-llm-api
cd clinical-llm-api
docker build -t clinical-llm-api .
docker run --gpus all -p 8000:8000 clinical-llm-api
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Please answer with one of the option in the bracket",
    "question": "Q: Which drug inhibits alpha-glucosidase?",
    "options": {"A": "Insulin", "B": "Metformin", "C": "Acarbose", "D": "Glyburide"}
  }'
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Base model | Llama 3.2 3B Instruct |
| Fine-tuning | QLoRA (Unsloth + PEFT + TRL) |
| Dataset | medalpaca/medical_meadow_medqa |
| Experiment tracking | WandB |
| Evaluation | Accuracy + ROUGE-L (HuggingFace evaluate) |
| Inference serving | FastAPI + Uvicorn |
| Containerization | Docker |
| Demo | Gradio on HuggingFace Spaces |
| Hardware | Google Colab L4 GPU |

---

## License

This project is a fine-tune of Meta's Llama 3.2 3B Instruct and is subject to the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE).
