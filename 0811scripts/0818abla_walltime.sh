export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_70B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='6'
export HF_ALLOW_CODE_EVAL=1
para=False

win_len=6
thre=0.3
# use_ngram=True  # True False
use_rede=True

max_ngram_size=4
num_ngram_pred_tokens=6

data=0819abla
for use_ngram in True False
do
for task in humaneval_instruct
do
    python -m lm_eval \
        --model hf \
        --model_args use_ngram=${use_ngram},max_ngram_size=${max_ngram_size},num_ngram_pred_tokens=${num_ngram_pred_tokens},entropy_thre=${thre},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=15,revise_decoding=${use_rede},win_len=${win_len} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_walltime_rede_70B_spdtok15_${task}_ngram${use_ngram}
done
done
