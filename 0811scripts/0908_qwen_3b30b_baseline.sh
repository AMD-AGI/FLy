export HF_HOME=/workspace/cache

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
llama32_1B='meta-llama/Llama-3.2-1B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B-Instruct-2507"
qwen3b30b="Qwen/Qwen3-30B-A3B-Instruct-2507"
qwen2_7b="Qwen/Qwen2-7B-Instruct"
mistral_7b="mistralai/Mistral-7B-Instruct-v0.3"
qwen25_7b="Qwen/Qwen2.5-7B-Instruct"
qwen25_72b="Qwen/Qwen2.5-72B-Instruct"
qwen25_math_7b="Qwen/Qwen2.5-Math-7B-Instruct"

draft_model=${qwen25_math_7b}
target_model=${qwen25_math_7b}


export HIP_VISIBLE_DEVICES='4'
export HF_ALLOW_CODE_EVAL=1
para=False

data=0912
model_name=baseline_qwen25_math_7b

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
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_3_${model_name}_${task}
done

# mgsm_native_cot_zh_0813 mgsm_native_cot_de_0813 mgsm_native_cot_en_0813 mgsm_native_cot_es_0813 mgsm_native_cot_fr_0813 mgsm_native_cot_ja_0813 mgsm_native_cot_ru_0813 mgsm_native_cot_sw_0813 mgsm_native_cot_te_0813 mgsm_native_cot_th_0813 mgsm_native_cot_bn_0813

