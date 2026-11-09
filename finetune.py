"""Optional real LoRA workflow; never used for numerical model training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import dump_json, now, utc
from .rag import TransformersLLM, prompt_for, validate_explanation


def make_examples(records, output, training_game_ids, training_cutoff):
    """Records must come from archived training-period forecast/evidence bundles."""
    examples = []
    for record in records:
        forecast, evidence = record["forecast"], record["evidence"]
        if forecast["game_id"] not in training_game_ids:
            raise ValueError("Fine-tuning game outside training partition")
        cutoff = utc(forecast["data_cutoff"])
        if cutoff > utc(training_cutoff):
            raise ValueError("Example beyond training period")
        for doc in evidence:
            for key in ("available_at", "published_at", "retrieved_at"):
                if not doc.get(key) or utc(doc[key]) > cutoff:
                    raise ValueError("Evidence unavailable at example cutoff")
        completion = record["explanation"].copy()
        completion.pop("provider_status", None)
        validate_explanation(json.dumps(completion), forecast, evidence)
        examples.append({"game_id": forecast["game_id"], "data_cutoff": str(cutoff),
                         "prompt": prompt_for(forecast, evidence),
                         "completion": json.dumps(completion)})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(x)+"\n" for x in examples))
    return len(examples)


def train_lora(dataset, output, config_path):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    config = json.loads(Path(config_path).read_text())
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["revision"])
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty fine-tuning dataset")

    class Examples(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            row = rows[i]
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                    tokenize=False, add_generation_prompt=True)
            text = prefix + row["completion"] + tokenizer.eos_token
            enc = tokenizer(text, truncation=True, max_length=config["max_length"],
                            padding="max_length", return_tensors="pt")
            enc = {k: v[0] for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            labels[:prefix_len] = -100
            labels[enc["attention_mask"] == 0] = -100
            if (labels != -100).sum() == 0:
                raise ValueError("Prompt consumes context; increase max_length or shorten evidence")
            enc["labels"] = labels
            return enc

    args = TrainingArguments(output_dir=str(output), num_train_epochs=config["epochs"],
                             per_device_train_batch_size=1, gradient_accumulation_steps=8,
                             learning_rate=2e-4, report_to=[], save_strategy="epoch",
                             use_cpu=not torch.cuda.is_available() and not torch.backends.mps.is_available(),
                             seed=config["seed"], logging_steps=5)
    trainer = Trainer(model=model, args=args, train_dataset=Examples())
    trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    dump_json(Path(output)/"checkpoint_metadata.json", {
        "completed_at": now(), "config": config, "base_resolved_commit": getattr(model.config, "_commit_hash", None),
        "examples": len(rows), "latest_example_cutoff": max(r["data_cutoff"] for r in rows),
        "training_game_ids": sorted({r["game_id"] for r in rows}),
        "device": str(next(model.parameters()).device), "license": "See base model license; adapter code MIT",
        "numerical_forecasting_improvement": "not tested; adapter only generates explanations"})


def compare_llms(bundles, output, base_model, adapter):
    """Measure raw generations before fallback; report both validity and groundedness."""
    metadata = json.loads((Path(adapter)/"checkpoint_metadata.json").read_text())
    if not bundles:
        raise ValueError("Need nonempty disjoint evaluation bundles")
    if base_model != metadata["config"]["base_model"]:
        raise ValueError("Comparison base must match adapter base model")
    train_games = set(metadata["training_game_ids"])
    if any(r["forecast"]["game_id"] in train_games for r in bundles):
        raise ValueError("LLM evaluation overlaps training games")
    if any(utc(r["forecast"]["data_cutoff"]) <= utc(metadata["latest_example_cutoff"]) for r in bundles):
        raise ValueError("LLM evaluation must follow fine-tuning examples")
    results = {}
    for name, path in [("base_rag", None), ("lora_rag", adapter)]:
        provider = TransformersLLM(base_model, path)
        counts = {"n": len(bundles), "structured_valid": 0, "numerical_preserved": 0,
                  "claims": 0, "valid_citations": 0, "unsupported_claims": 0, "fully_valid": 0}
        generations = []
        for row in bundles:
            raw = provider.generate(prompt_for(row["forecast"], row["evidence"]))
            generations.append(raw)
            try:
                parsed = json.loads(raw)
                from .rag import Explanation
                parsed = Explanation.model_validate(parsed)
                counts["structured_valid"] += 1
                counts["numerical_preserved"] += int(parsed.forecast == row["forecast"])
                sources = {x["source_id"]: x["text"] for x in row["evidence"]}
                for claim in parsed.claims:
                    counts["claims"] += 1
                    valid_ids = all(x in sources for x in claim.source_ids)
                    counts["valid_citations"] += int(valid_ids)
                    grounded = valid_ids and any(claim.text in sources[x] for x in claim.source_ids)
                    counts["unsupported_claims"] += int(not grounded)
                validate_explanation(raw, row["forecast"], row["evidence"])
                counts["fully_valid"] += 1
            except (ValueError, TypeError, KeyError):
                pass
        rates = {"structured_valid_rate": counts["structured_valid"]/max(1, counts["n"]),
                 "numerical_preservation_rate": counts["numerical_preserved"]/max(1, counts["n"]),
                 "citation_correctness_rate": counts["valid_citations"]/max(1, counts["claims"]),
                 "unsupported_claim_rate": counts["unsupported_claims"]/max(1, counts["claims"])}
        results[name] = counts | rates | {"raw_generations": generations}
        del provider
    dump_json(output, results | {"forecasting_improvement": "not evaluated; numerical model unchanged"})
    return results


def generate_grounded_bundles(root, partitions, output, max_examples=80):
    """Generate extraction supervision from strictly timestamped training evidence.

    Uses a no-fit historical mean baseline for supplied numbers, not a numerical model
    trained after the example. Produces a disjoint later evaluation bundle alongside it.
    """
    from .rag import retrieve, explain
    from .model import eligible

    frame = eligible(partitions)
    bundles = {}
    for name in ("train", "test"):
        candidates = frame[frame.partition == name].sort_values("cutoff")
        if len(candidates) > max_examples:
            candidates = candidates.iloc[
                np.linspace(0, len(candidates)-1, max_examples).astype(int)]
        records = []
        for row in candidates.to_dict("records"):
            evidence = retrieve(root, Path(root)/"text_index.parquet", row["player_id"],
                                row["team_id"], row["opponent_id"], row["season"], row["cutoff"])
            safe = []
            for doc in evidence:
                timestamps = [utc(doc.get(k)) for k in ("available_at", "published_at", "retrieved_at")]
                if all(not pd.isna(t) and t <= utc(row["cutoff"]) for t in timestamps):
                    safe.append(doc)
            if not safe:
                continue
            forecast = {"game_id": row["game_id"], "player_id": row["player_id"],
                        "data_cutoff": str(row["cutoff"]), "expected_points": round(row["season_points"], 4),
                        "median_points": round(row["points_20"], 4),
                        "availability_assumptions": ["Conditional on playing; availability unknown."],
                        "missing_information": ["Verified injuries and starting role are unavailable."],
                        "supervision_note": "No-fit historical baseline supplies immutable example numbers; not a primary model forecast."}
            record = {"forecast": forecast, "evidence": safe[:3]}
            record["explanation"] = explain(forecast, safe[:3])
            records.append(record)
        bundles[name] = records
    if not bundles["train"]:
        raise ValueError("No verified timestamped training evidence. Use synthetic demo or archived training-period bundles; date-proxy real logs are insufficient for LLM training.")
    train_ids = set(frame.loc[frame.partition == "train", "game_id"].astype(str))
    make_examples(bundles["train"], output, train_ids, frame.loc[frame.partition == "train", "cutoff"].max())
    evaluation_path = Path(output).parent/"evaluation_bundles.json"
    dump_json(evaluation_path, bundles["test"])
    dump_json(Path(output).parent/"dataset_metadata.json", {
        "created_at": now(), "training_examples": len(bundles["train"]),
        "evaluation_examples": len(bundles["test"]),
        "synthetic": bool(frame.synthetic.all()),
        "supervision": "Extractive renderer targets; no human reasoning labels; immutable baseline numerical context",
        "evaluation_path": str(evaluation_path)})
    return len(bundles["train"])
