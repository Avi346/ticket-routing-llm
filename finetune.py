import json

import data


def training_messages(name, seed, budget):
    train = data.load_train(name, seed, budget)
    rows = []

    for text, label in zip(train["text"], train["label"]):
        rows.append({
            "messages": [
                {"role": "user", "content": f"Message: {text}"},
                {"role": "assistant", "content": json.dumps({"category": label})},
            ]
        })

    return rows

def load_model(seed):
    from unsloth import FastLanguageModel
    from transformers import set_seed

    import config

    set_seed(seed)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=config.LLM_MODEL,
        max_seq_length=config.LORA_MAX_LEN,
        load_in_4bit=False,
        load_in_16bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    tok = getattr(tok, "tokenizer", tok)
    tok.padding_side = "right"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok

def training_dataset(name, seed, budget, tok):
    from datasets import Dataset

    import config
    import prompts

    rows = []

    for row in training_messages(name, seed, budget):
        messages = row["messages"]

        prompt = tok.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        if config.USE_ANSWER_PREFIX:
            prompt += prompts.ANSWER_PREFIX

        full = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        assert full.startswith(prompt), "Training and generation formats differ"

        rows.append({
            "prompt": prompt,
            "completion": full[len(prompt):],
        })

    return Dataset.from_list(rows)

def train_lora(model, tok, train_ds, seed, folder):
    import math
    import torch
    from trl import SFTConfig, SFTTrainer

    import config

    if (folder / "adapter_config.json").exists():
        raise FileExistsError("Adapter already saved. Stop before retraining.")

    batch = config.LORA_BATCH * config.LORA_ACCUM
    steps = max(
        config.LORA_EPOCHS * math.ceil(len(train_ds) / batch),
        config.LORA_MIN_STEPS,
    )

    args = SFTConfig(
        output_dir=str(folder),
        max_steps=steps,
        max_length=config.LORA_MAX_LEN,
        per_device_train_batch_size=config.LORA_BATCH,
        gradient_accumulation_steps=config.LORA_ACCUM,
        learning_rate=config.LORA_LR,
        warmup_steps=steps // 10,
        optim="adamw_8bit",
        fp16=True,
        bf16=False,
        packing=False,
        completion_only_loss=True,
        dataset_num_proc=1,
        seed=seed,
        data_seed=seed,
        logging_steps=25,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        args=args,
    )

    sample = trainer.data_collator([trainer.train_dataset[0]])
    labels = sample["labels"][0]
    learned = tok.decode(
        labels[labels != -100].tolist(), skip_special_tokens=True
    ).strip()
    expected = train_ds[0]["completion"].replace(tok.eos_token, "").strip()
    assert learned == expected, "Training target mismatch"

    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()

    if not math.isfinite(result.training_loss):
        raise RuntimeError("Non-finite training loss. Adapter not saved.")

    trainer.save_model(str(folder))
    tok.save_pretrained(str(folder))

    stats = {
        "steps": result.global_step,
        "seconds": result.metrics["train_runtime"],
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "train_loss": result.training_loss,
    }
    (folder / "training_stats.json").write_text(json.dumps(stats, indent=2))

    print("Peak GPU GB:", round(stats["peak_gpu_gb"], 2))
    print("Saved:", folder)
    return trainer

def predict(model, tok, texts):
    from unsloth import FastLanguageModel
    import torch

    import config
    import prompts

    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"

    prefix = prompts.ANSWER_PREFIX if config.USE_ANSWER_PREFIX else ""
    chats = []

    for text in texts:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": f"Message: {text}"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chats.append(chat + prefix)

    raw, n_in, n_out = [], [], []
    batch = config.BATCH_SIZE["zero"]

    for i in range(0, len(chats), batch):
        inputs = tok(
            chats[i:i + batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")

        with torch.inference_mode():
           out = model.generate(
    **inputs,
    max_new_tokens=config.MAX_NEW_TOKENS,
    do_sample=False,
    use_cache=True,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)

        new = out[:, inputs["input_ids"].shape[1]:]
        replies = tok.batch_decode(new, skip_special_tokens=True)

        raw.extend(prefix + reply.strip() for reply in replies)
        n_in.extend(inputs["attention_mask"].sum(1).tolist())
        n_out.extend((new != tok.pad_token_id).sum(1).tolist())

    return raw, n_in, n_out


def train_run(name, seed, budget):
    import config

    folder = config.ADAPTER_DIR / name / f"b{budget}_s{seed}"

    if (folder / "training_stats.json").exists():
        print("Already trained:", folder)
        return

    model, tok = load_model(seed)
    train_ds = training_dataset(name, seed, budget, tok)

    train_lora(model, tok, train_ds, seed, folder)