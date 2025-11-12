export HF_HOME=/workspace/cache

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B"
qwen3_1_7b="Qwen/Qwen3-1.7B"
qwen72b='Qwen/Qwen2.5-72B-Instruct'

dsr1="/data/DeepSeek-R1"

draft_model=${qwen3_1_7b}
target_model=${dsr1}


export HIP_VISIBLE_DEVICES='0,1,2,3'
export HF_ALLOW_CODE_EVAL=1
para=True

data=0821dsr1


for task in humaneval_instruct
do
    python -m lm_eval \
        --model hf \
        --model_args pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_2_dsr1_${task}_test
done

