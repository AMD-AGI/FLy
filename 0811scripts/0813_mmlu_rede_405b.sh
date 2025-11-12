export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_405B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='0,1,2,3'
para=True

data=0816

win_len=6
thre=0.8
use_ngram=True  # True False
use_rede=True

max_ngram_size=4
num_ngram_pred_tokens=6


for lang in de fr es th zh
do
# for subject in computer_science chemistry business math law biology economics engineering history health other philosophy physics psychology
for subject in computer_science chemistry business math engineering physics
do
    task=mmlu_prox_${lang}_${subject}
    python -m lm_eval \
        --model hf \
        --model_args use_ngram=${use_ngram},max_ngram_size=${max_ngram_size},num_ngram_pred_tokens=${num_ngram_pred_tokens},entropy_thre=${thre},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=25,revise_decoding=${use_rede},win_len=${win_len} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --limit 50 \
        &> ./output_log/${data}_6_405B_${task}_UseReDe${use_rede}_Usengram${use_ngram}_ngramSize${max_ngram_size}_ngramPredict${num_ngram_pred_tokens}_winL${win_len}
done
done

