#!/bin/env python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Meta-Llama-3-8B-Instruct")


sentence = "Hey how are you doing today?"
tokened_sentence = tokenizer(sentence, return_tensors="pt")
print(tokened_sentence)

#model.to("cuda")
print(model)
print(model(**tokened_sentence))

