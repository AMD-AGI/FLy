export HF_HOME=/workspace/cache

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B"

draft_model=${llama31_8B}
target_model=${llama31_405B}


export HIP_VISIBLE_DEVICES='0,1,2,3'
export HF_ALLOW_CODE_EVAL=1
para=True

temp=1
data=0822
# task=humaneval_instruct

for task in humaneval_instruct mgsm_native_cot_th_0813 mbpp_plus_instruct

do
    python -m lm_eval \
        --model hf \
        --model_args temperature=${temp},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_3_Baseline405b_${task}_temp${temp}_alldata
done

# humaneval_instruct acp_prog_gen gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 mgsm_native_cot_th_0813 mbpp_plus_instruct niah_multivalue
