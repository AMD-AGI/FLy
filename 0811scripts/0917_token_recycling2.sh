# export HF_HOME=/workspace/cache
export HF_HOME=/data

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
llama32_1B='meta-llama/Llama-3.2-1B-Instruct'

target_model=${llama31_405B}
draft_model=${llama32_1B}

export HIP_VISIBLE_DEVICES='0,5,6,7'
export HF_ALLOW_CODE_EVAL=1
para=True

# temp=0

data=0917_TR

for temp in 0 1
do
    for task in humaneval_instruct mgsm_native_cot_th_0813 mbpp_plus_instruct acp_prog_gen gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 niah_multivalue
    do
        python -m lm_eval \
            --model hf \
            --model_args TokenRecycling=True,temperature=${temp},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model} \
            --tasks ${task} \
            --output_path ./eval_out/${data}_${task} \
            --show_config \
            --log_samples \
            --batch_size 1 \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code \
            --limit 80 \
            &> ./output_log/${data}_405B_${task}_temp${temp}
    done
done
# humaneval_instruct mgsm_native_cot_th_0813 mbpp_plus_instruct
# acp_prog_gen gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 niah_multivalue