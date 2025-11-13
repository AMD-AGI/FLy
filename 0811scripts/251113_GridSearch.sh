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
export HIP_VISIBLE_DEVICES='0'
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

# Grid search parameters
spd_k_values=(10 15 20 25)
win_len_values=(4 6 8)
entropy_thre_values=(0.3 0.6)

# Counter for tracking experiments
exp_count=0
total_exps=$((${#spd_k_values[@]} * ${#win_len_values[@]} * ${#entropy_thre_values[@]}))

echo "Starting Grid Search: ${total_exps} experiments"
echo "spd_k: ${spd_k_values[@]}"
echo "win_len: ${win_len_values[@]}"
echo "entropy_thre: ${entropy_thre_values[@]}"
echo "=========================================="

# Grid search: nested loops
for spd_k in "${spd_k_values[@]}"; do
    for win_len in "${win_len_values[@]}"; do
        for entropy_thre in "${entropy_thre_values[@]}"; do
            exp_count=$((exp_count + 1))
            
            # Create unique identifier for this experiment
            exp_id="k${spd_k}_w${win_len}_e${entropy_thre}"
            
            echo ""
            echo "[${exp_count}/${total_exps}] Running experiment: spd_k=${spd_k}, win_len=${win_len}, entropy_thre=${entropy_thre}"
            echo "Experiment ID: ${exp_id}"
            
            # Run evaluation with current parameter combination
            python -m lm_eval \
                --model hf \
                --model_args pretrained=${draft_model},parallelize=${para},config_path=${config_path},spd_k=${spd_k},win_len=${win_len},entropy_thre=${entropy_thre} \
                --tasks ${task} \
                --output_path ./eval_out/${data}_${task}_${exp_id} \
                --show_config \
                --log_samples \
                --batch_size 1 \
                --apply_chat_template \
                --fewshot_as_multiturn \
                --confirm_run_unsafe_code \
                &> ./output_log/${data}_${task}_${exp_id}.log
            
            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "[${exp_count}/${total_exps}] ✓ Experiment ${exp_id} completed successfully"
            else
                echo "[${exp_count}/${total_exps}] ✗ Experiment ${exp_id} failed (check log: ./output_log/${data}_${task}_${exp_id}.log)"
            fi
            
            echo "------------------------------------------"
        done
    done
done

echo ""
echo "=========================================="
echo "Grid Search Complete! Total experiments: ${total_exps}"
echo "Results saved in: ./eval_out/${data}_${task}_*"
echo "Logs saved in: ./output_log/${data}_${task}_*.log"

