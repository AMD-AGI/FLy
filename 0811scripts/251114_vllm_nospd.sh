#!/bin/bash

export HF_HOME=/wekafs/jinzeli2/cache

# Model paths
llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

# Target model (main model)
target_model=${llama31_70B}

# GPU configuration
export CUDA_VISIBLE_DEVICES='0'
# export HIP_VISIBLE_DEVICES='4'
export HF_ALLOW_CODE_EVAL=1

# FLy algorithm parameters
enable_fly=False
fly_win_len=6
entropy_threshold=0.3  # Set to a float value if needed, e.g., 0.5

# Task to evaluate
task=gsm8k_cot_llama

# Output directory
data=251115

# Run evaluation with vLLM
for bs in 1
do
    python -m lm_eval \
        --model vllm \
        --model_args pretrained=${target_model} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_vllm_${task} \
        --show_config \
        --log_samples \
        --batch_size $bs \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        --limit 200 \
        &> ./output_log/${data}_${task}_vllm_FLy_nospd_bs${bs}_tp1.log
done
