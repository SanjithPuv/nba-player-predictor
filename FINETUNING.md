# Optional explanation LoRA experiment

## What is implemented and what ran

Implemented: training-only evidence examples, completion-masked causal training, LoRA adapters, saved model/revision/device/example metadata, and disjoint later-game base-plus-RAG versus adapter-plus-RAG evaluation. No model weights were downloaded, no LoRA optimization ran, and no base/adapter score was measured during the core project build. There is no claim that explanation fine-tuning improves points forecasts.

The synthetic grounded-example generator can run without Torch or a GPU. It uses archived synthetic pregame evidence and immutable historical baseline numbers, not the eventual model's fitted training predictions. A renderer provides extractive target explanations. This teaches schema following and quotation/citation behavior, not general basketball reasoning. The separate test examples have later cutoffs and different target game IDs. Test examples can reference earlier historical evidence, as a real forecast would.

## Commands

First run the synthetic demo in the README. Then:

```bash
nba-predictor --root data/synthetic --artifacts artifacts/synthetic llm-examples --output data/llm/train.jsonl
python -m pip install '.[llm]'
python -m pip freeze > artifacts/llm-environment.txt
nba-predictor finetune data/llm/train.jsonl --output artifacts/lora --lora-config config/lora.json
nba-predictor llm-evaluate data/llm/evaluation_bundles.json --adapter artifacts/lora
```

Before installing optional dependencies, choose a supported Python/hardware stack. Torch wheels can lag new Python releases. The core forecasting environment does not depend on Torch. A Python 3.11 or 3.12 separate environment is a sensible option if Torch is unavailable for your current interpreter. It needs the project and optional dependencies installed; the core Python-3.14 lock is not a Torch lock.

For genuine historical reports, supply a JSON array of archived bundles, each containing `forecast`, `evidence`, and a validated target `explanation`. Use `llm-examples bundles.json`. Each target game must be in the numerical model's training partition, each example cutoff must be in the training period, and publication, observed retrieval and availability for every evidence item must precede that example's cutoff. Real logs downloaded today with missing historical publication are not sufficient for this workflow; they are intentionally rejected. Do not relabel today's retrieval as a historical one.

## Model, licensing, and hardware

Default: `Qwen/Qwen2.5-0.5B-Instruct`, a roughly half-billion-parameter instruction model. Its [publisher model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) declares Apache-2.0. Review the exact model license and download terms. Replace `revision: main` in `config/lora.json` with a verified commit before a reproducible research run; resolved base commit is recorded as well. Save an exact optional environment for each run. Network access/model download and storage are required unless weights are already cached.

LoRA rank 8, alpha 16, dropout .05, query/value projections; batch 1, accumulation 8, learning rate 0.0002, two epochs, sequence length 4096. Only completion tokens contribute loss; instructions and padding are masked. A 4096-token sequence can consume substantial activation memory even for a small model. Approximately 8–16 GB accelerator memory or 16+ GB system RAM is a planning estimate, not a measured requirement. CUDA, Apple MPS, and slow CPU fallback depend on installed Torch support. No quantization dependency is required. Reduce sequence length/evidence size if needed; examples that truncate away the completion fail explicitly rather than silently training on padding.

## Evaluation meaning

`llm_comparison.json` records both raw generations and counts for structured validity, exact preservation of forecast numbers, citation IDs, grounded excerpts, and complete validity. Citation correctness means the claimed ID was retrieved; groundedness means the exact claim text occurs in a cited excerpt. Neither proves the original document is correct. An empty claim list can game these metrics: inspect claim counts and raw generations as well as validity. For research-quality comparison, add human judgments of relevance, harmful omission and interpretation, using a fixed rubric and blinded order.

The evaluation rejects training-game overlap and cutoffs before the last training example. These checks establish example separation, not the base model's own historical knowledge boundary. An actual prospective comparison must capture the adapter creation timestamp and base revision before the evaluation games occur. Comparing on already finished games is an explanation benchmark only. No LLM is used to fabricate missing historical injuries or revise numerical predictions.
