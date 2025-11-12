export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_8B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='1'
export HF_ALLOW_CODE_EVAL=1
para=False

temp=0

data=0917sb
method="recycling"  # recycling

for task in humaneval_instruct
do
    python -m lm_eval \
        --model hf \
        --model_args spec_bench_method=${method},temperature=${temp},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        --limit 8 \
        &> ./output_log/${data}_70B_${task}_${method}_temp${temp}_test
done
