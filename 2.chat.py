#!/bin/env python

# Based on https://github.com/meta-llama/llama3/blob/main/example_chat_completion.py

from typing import List, Optional
from llama import Dialog, Llama
import os

os.environ["MASTER_ADDR"]="127.1"
os.environ["MASTER_PORT"]="13456"
os.environ["RANK"]="0"
os.environ["WORLD_SIZE"]="1"


ckpt_dir: str = "Meta-Llama-3-8B-Instruct/original/"
tokenizer_path: str = "Meta-Llama-3-8B-Instruct/original/tokenizer.model"
temperature: float = 0.6
top_p: float = 0.9
max_seq_len: int = 512
max_batch_size: int = 4
max_gen_len: Optional[int] = None

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)

dialogs: List[Dialog] = [
    [],
]

while True:
    prompt = input("Q: ")

    dialogs[0].append({"role": "user", "content": prompt})

    max_gen_len = generator.model.params.max_seq_len - 1

    prompt_tokens = [
        generator.formatter.encode_dialog_prompt(dialogs[0])
    ]

    print("A: ", end = "")
    generation_tokens, generation_logprobs = generator.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        output_token__one_by_one = True,
    )
    print("")

    response = generator.tokenizer.decode(generation_tokens[0])

    dialogs[0].append({
                    "role": "assistant",
                    "content": response,
                })

