# export HF_HOME=/workspace/cache
# llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
# llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
# llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
# llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
# llama32_1B='meta-llama/Llama-3.2-1B-Instruct'
# qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
# qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
# qwen3_32b="Qwen/Qwen3-32B"
# qwen3_4b="Qwen/Qwen3-4B-Instruct-2507"
# qwen3b30b="Qwen/Qwen3-30B-A3B-Instruct-2507"
# qwen2_7b="Qwen/Qwen2-7B-Instruct"
# mistral_8b="mistralai/Ministral-8B-Instruct-2410"
# qwen25_7b="Qwen/Qwen2.5-7B-Instruct"
# mistral_large="mistralai/Mistral-Large-Instruct-2411"

export HF_HOME=/data
# dsr1="deepseek-ai/DeepSeek-R1-0528"
llama32_1B='meta-llama/Llama-3.2-1B-Instruct'
llama33_70B='meta-llama/Llama-3.1-70B-Instruct'
dsr1_llama70b="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
dsr1_qwen_15b="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dsr1_llama_8b="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
dsr1_qwen_8b="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

draft_model=${llama32_1B}
target_model=${dsr1_qwen_8b}


export HIP_VISIBLE_DEVICES='7'
export HF_ALLOW_CODE_EVAL=1
para=False

data=0914
model_name=dsr1_qwen_8b

# gsm8k_cot_llama humaneval_instruct
for task in humaneval_instruct
do
    python -m lm_eval \
        --model hf \
        --model_args verbose=False,pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_2_${model_name}_${task}
done

# mgsm_native_cot_zh_0813 mgsm_native_cot_de_0813 mgsm_native_cot_en_0813 mgsm_native_cot_es_0813 mgsm_native_cot_fr_0813 mgsm_native_cot_ja_0813 mgsm_native_cot_ru_0813 mgsm_native_cot_sw_0813 mgsm_native_cot_te_0813 mgsm_native_cot_th_0813 mgsm_native_cot_bn_0813

        # --apply_chat_template \
        # --fewshot_as_multiturn \
