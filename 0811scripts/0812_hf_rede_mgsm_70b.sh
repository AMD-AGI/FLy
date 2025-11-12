export HF_HOME=/shareddata/jinzeli2/cache

# target_model=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
# draft_model=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_405B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='4,5,6,7'
para=True

win_len=8
thre=0.3
use_ngram=True  # True False
use_rede=True

max_ngram_size=4
num_ngram_pred_tokens=6


for use_ngram in True
do
for task in mgsm_native_cot_de mgsm_native_cot_es mgsm_native_cot_fr mgsm_native_cot_ru mgsm_native_cot_th mgsm_native_cot_zh
do
    python -m lm_eval \
        --model hf \
        --model_args use_ngram=${use_ngram},max_ngram_size=${max_ngram_size},num_ngram_pred_tokens=${num_ngram_pred_tokens},entropy_thre=${thre},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=25,revise_decoding=${use_rede},win_len=${win_len},total_gen_tok=2048 \
        --tasks ${task} \
        --output_path ./eval_out/0812_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        &> ./output_log/0813_1_HF405B_${task}_UseReDe${use_rede}_Usengram${use_ngram}_ngramSize${max_ngram_size}_ngramPredict${num_ngram_pred_tokens}_winL${win_len}
done
done

