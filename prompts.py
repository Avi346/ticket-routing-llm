import json

ANSWER_PREFIX = '{"category": "'     # only used if the dev test says it helps


def build_prompt(text, labels, shots=()):
    parts = ["Classify the customer message into exactly one of these categories:",
             "\n".join(labels), ""]
    if shots:
        parts.append("Examples:")
        for t, l in shots:
            parts.append(f"Message: {t}\nAnswer: {json.dumps({'category': l})}")
        parts.append("")
    parts.append(f"Message: {text}")
    parts.append('Reply with JSON only, like {"category": "<label>"}.')
    return "\n".join(parts)