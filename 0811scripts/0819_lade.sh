export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

lade_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='3'
export HF_ALLOW_CODE_EVAL=1
para=False

use_lade=True

data=0819lade
for task in acp_prog_gen 
do
    python -m lm_eval \
        --model hf \
        --model_args pretrained=${lade_model},parallelize=${para},use_lade=${use_lade} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_405B_${task}_winl${win_len}
done

# acp_prog_gen humaneval_instruct gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 mgsm_native_cot_th_0813 mbpp_plus_instruct niah_multivalue
