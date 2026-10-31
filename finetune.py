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