export HF_HOME=/workspace/cache

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B"

draft_model=${qwen3_4b}
target_model=${qwen3_32b}


export HIP_VISIBLE_DEVICES='5'
export HF_ALLOW_CODE_EVAL=1
para=False

data=0821qwen


for task in humaneval_instruct
do
    python -m lm_eval \
        --model hf \
        --model_args enable_qwen_thinking=False,pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_2_Baseline32b_${task}
done

# mgsm_native_cot_zh_0813 mgsm_native_cot_de_0813 mgsm_native_cot_en_0813 mgsm_native_cot_es_0813 mgsm_native_cot_fr_0813 mgsm_native_cot_ja_0813 mgsm_native_cot_ru_0813 mgsm_native_cot_sw_0813 mgsm_native_cot_te_0813 mgsm_native_cot_th_0813 mgsm_native_cot_bn_0813

