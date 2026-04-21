import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


BASE_MODEL = "distilgpt2"
OUTPUT_DIR = Path("artifacts/trl_sft_checkpoint")
METRICS_PATH = Path("artifacts/trl_sft_training_metrics.json")


def _build_examples() -> list[dict]:
    prompt_1 = (
        "Dataset has columns: contact_id, email, phone, country. Missing values are high in email and phone. "
        "Choose the best next action."
    )
    action_1 = {
        "action_type": "analyze",
        "target_columns": ["email", "phone"],
        "parameters": {},
        "reasoning": "Start by analyzing highest-missing columns before imputation."
    }

    prompt_2 = (
        "Duplicate rows were detected in invoice records. Columns include invoice_id, account_id, amount, status."
    )
    action_2 = {
        "action_type": "deduplicate",
        "target_columns": ["invoice_id", "account_id"],
        "parameters": {"subset": ["invoice_id"], "keep": "first"},
        "reasoning": "Remove duplicated invoice rows while preserving first valid entry."
    }

    prompt_3 = (
        "Support tickets include numeric csat_score and some missing values. Validate quality after cleaning."
    )
    action_3 = {
        "action_type": "validate",
        "target_columns": ["csat_score"],
        "parameters": {"csat_score_type": "numeric", "csat_score_min": 1, "csat_score_max": 5},
        "reasoning": "Ensure CSAT values stay within allowed numeric range after cleaning."
    }

    examples = []
    for prompt, action in [(prompt_1, action_1), (prompt_2, action_2), (prompt_3, action_3)]:
        text = (
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
            f"{json.dumps(action)}"
        )
        examples.append({"text": text})
    return examples


def main() -> None:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_list(_build_examples())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=False,
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics = {
        "base_model": BASE_MODEL,
        "train_examples": len(dataset),
        "train_runtime": result.metrics.get("train_runtime"),
        "train_steps_per_second": result.metrics.get("train_steps_per_second"),
        "train_loss": result.metrics.get("train_loss"),
        "global_step": result.metrics.get("global_step"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
