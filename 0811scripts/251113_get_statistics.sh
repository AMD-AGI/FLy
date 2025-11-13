#!/bin/bash

export HF_HOME=/wekafs/jinzeli2/cache

# Model paths
llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

# Draft model (smaller model for speculation)
draft_model=${llama31_8B}

# GPU configuration
export HIP_VISIBLE_DEVICES='1'
# export HIP_VISIBLE_DEVICES='4'
export HF_ALLOW_CODE_EVAL=1

# Parallelization
para=False

# JSON config file path (relative to project root)
config_path="fly/FLy_Llama3_70b.json"

# Task to evaluate
task=humaneval_instruct

# Output directory
data=251113

# Run evaluation with statistics enabled
python -m lm_eval \
    --model hf \
    --model_args pretrained=${draft_model},parallelize=${para},config_path=${config_path},enable_statistics=True \
    --tasks ${task} \
    --output_path ./eval_out/${data}_${task}_statistics \
    --show_config \
    --log_samples \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code \
    &> ./output_log/${data}_${task}_FLy_Llama3_70b_statistics.log

