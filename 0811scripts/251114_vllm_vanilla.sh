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
task=humaneval_instruct

# Output directory
data=251114

# Run evaluation with vLLM
python -m lm_eval \
    --model vllm \
    --model_args pretrained=${target_model},enable_fly=${enable_fly},fly_win_len=${fly_win_len},entropy_threshold=${entropy_threshold},spd_k=15,draft_model=${llama31_8B} \
    --tasks ${task} \
    --output_path ./eval_out/${data}_vllm_${task} \
    --show_config \
    --log_samples \
    --batch_size 16 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code \
    &> ./output_log/${data}_${task}_vllm_FLy_bs16_sps2.log

