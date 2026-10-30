import unsloth  # must be imported before transformers
from unsloth import FastLanguageModel
import time
import data
import evaluate
import hashlib
import json
import os
import shutil
import sqlite3

import torch

import config
import prompts

model = tok = None


def load_model():
    global model, tok
    model, tok = FastLanguageModel.from_pretrained(
        model_name=config.LLM_MODEL, max_seq_length=config.MAX_SEQ_LEN,
        load_in_4bit=False, load_in_16bit=True)
    FastLanguageModel.for_inference(model)
    tok = getattr(tok, "tokenizer", tok)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token



def to_chat(user_text, use_prefix=False):
    msgs = [{"role": "user", "content": user_text}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return chat + prompts.ANSWER_PREFIX if use_prefix else chat


class Cache:
    def __init__(self):
        config.CACHE_DRIVE.parent.mkdir(exist_ok=True)
        if config.CACHE_DRIVE.exists() and not os.path.exists(config.CACHE_LOCAL):
            shutil.copy(config.CACHE_DRIVE, config.CACHE_LOCAL)
        self.con = sqlite3.connect(config.CACHE_LOCAL)
        self.con.execute("CREATE TABLE IF NOT EXISTS gen "
                         "(key TEXT PRIMARY KEY, output TEXT, n_in INT, n_out INT)")
        self.con.commit()

    def key(self, chat, bs):
        blob = json.dumps({"model": config.LLM_MODEL, "prompt": chat,
                           "max_new_tokens": config.MAX_NEW_TOKENS, "batch": bs},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, chat, bs):
        return self.con.execute("SELECT output, n_in, n_out FROM gen WHERE key=?",
                                (self.key(chat, bs),)).fetchone()

    def put(self, chat, bs, output, n_in, n_out):
        self.con.execute("INSERT OR REPLACE INTO gen VALUES (?,?,?,?)",
                         (self.key(chat, bs), output, n_in, n_out))

    def count(self):
        return self.con.execute("SELECT COUNT(*) FROM gen").fetchone()[0]

    def backup(self):
        self.con.commit()
        shutil.copy(config.CACHE_LOCAL, config.CACHE_DRIVE)


def generate_batch(chats):
    inputs = tok(chats, return_tensors="pt", padding=True,
                 add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    width = inputs["input_ids"].shape[1]
    n_in = inputs["attention_mask"].sum(1).tolist()
    results = []
    for i, o in enumerate(out):
        new = o[width:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        results.append((text, n_in[i], int((new != tok.pad_token_id).sum())))
    return results


def run(user_texts, kind, cache, use_prefix=False):
    bs = config.BATCH_SIZE[kind]
    chats = [to_chat(u, use_prefix) for u in user_texts]
    todo = [c for c in dict.fromkeys(chats) if cache.get(c, bs) is None]
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        for chat, r in zip(batch, generate_batch(batch)):
            cache.put(chat, bs, *r)
        cache.con.commit()
        if (i // bs) % 25 == 0:
            cache.backup()
    cache.backup()
    rows = [cache.get(c, bs) for c in chats]
    prefix = prompts.ANSWER_PREFIX if use_prefix else ""
    return [prefix + r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def dev_check(name, kind, cache, use_prefix=False, shots=(), n=300):
    labels = data.load_labels(name)
    dev = data.load_dev_subsample(name).head(n)
    users = [prompts.build_prompt(t, labels, shots) for t in dev["text"]]
    t0 = time.time()
    raw, n_in, n_out = run(users, kind, cache, use_prefix)
    sec = (time.time() - t0) / len(users)
    pred = [evaluate.parse_output(r, labels) for r in raw]
    return {"acc": round(float(evaluate.accuracy(dev["label"].tolist(), pred)), 3),
            "valid": round(evaluate.json_validity(raw, labels), 3),
            "tok_in": round(sum(n_in) / len(n_in)), "tok_out": round(sum(n_out) / len(n_out), 1),
            "sec_per_item": round(sec, 2)}