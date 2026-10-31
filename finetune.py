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