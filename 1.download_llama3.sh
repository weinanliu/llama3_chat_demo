#!/bin/bash

set -x

#pip install huggingface-hub hf_transfer

export HF_ENDPOINT=https://hf-mirror.com

## https://huggingface.co/settings/tokens
TOKEN=hf_HEVqnbckgZBXbzGSsHgqiWrloetXIUbWY
USERNAME=einan9710
#huggingface-cli login --token ${TOKEN}

AUTHOR=meta-llama
#MODEL_NAME=Meta-Llama-3-8B
MODEL_NAME=Meta-Llama-3-8B-Instruct

bash ./hfd.sh \
         ${AUTHOR}/${MODEL_NAME} \
         --local-dir ${MODEL_NAME} \
         --hf_username ${USERNAME} \
         --hf_token ${TOKEN}

#huggingface-cli download \
#        --resume-download \
#        ${AUTHOR}/${MODEL_NAME} \
#        --local-dir ${MODEL_NAME} \
#        --local-dir-use-symlinks False \
