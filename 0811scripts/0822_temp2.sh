export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_405B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='4,5,6,7'
export HF_ALLOW_CODE_EVAL=1
para=True

win_len=8
thre=0.3
use_ngram=True  # True False
use_rede=True
task=humaneval_instruct

max_ngram_size=4
num_ngram_pred_tokens=6
temp=1
data=0822

# for win_len in 3 4 5 6 7 8 
# do
for task in humaneval_instruct mgsm_native_cot_th_0813 mbpp_plus_instruct

do
    python -m lm_eval \
        --model hf \
        --model_args temperature=${temp},use_ngram=${use_ngram},max_ngram_size=${max_ngram_size},num_ngram_pred_tokens=${num_ngram_pred_tokens},entropy_thre=${thre},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=25,revise_decoding=${use_rede},win_len=${win_len} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_2_rede_405B_${task}_temp${temp}_sample1times
done
# done

# humaneval_instruct acp_prog_gen gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 mgsm_native_cot_th_0813 mbpp_plus_instruct niah_multivalue
