#!/bin/bash
set -e

export HF_ALLOW_CODE_EVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pip install -e . 2>&1 | tail -5

DRAFT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
CONFIG_PATH="fly_config/FLy_Llama3_70b.json"

fly --model hf \
    --model_args pretrained=${DRAFT_MODEL},config_path=${CONFIG_PATH} \
    --tasks humaneval_instruct \
    --batch_size 1 \
    --confirm_run_unsafe_code
