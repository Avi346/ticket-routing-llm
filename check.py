import data, prompts

labels = data.load_labels("banking77")
train = data.load_train("banking77", 0, 2)
shots = list(zip(train["text"], train["label"]))
print(prompts.build_prompt("my card has not arrived", labels, shots[:2]))