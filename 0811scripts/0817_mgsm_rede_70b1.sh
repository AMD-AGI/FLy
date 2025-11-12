export HF_HOME=/workspace/cache

# target_model=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
# draft_model=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_70B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='2'
para=False

win_len=6
# thre=0.6
use_ngram=True  # True False
use_rede=True

max_ngram_size=4
num_ngram_pred_tokens=6

data=0817
for thre in 0.6 0.7
do
for task in mgsm_native_cot_de_0813 mgsm_native_cot_fr_0813 mgsm_native_cot_th_0813
do
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
        &> ./output_log/${data}_7_rede_70B_${task}_UseReDe${use_rede}_Usengram${use_ngram}_ngramSize${max_ngram_size}_ngramPredict${num_ngram_pred_tokens}_winL${win_len}_thre${thre}
done
done

# mgsm_native_cot_zh_0813 mgsm_native_cot_de_0813 mgsm_native_cot_en_0813 mgsm_native_cot_es_0813 mgsm_native_cot_fr_0813 mgsm_native_cot_ja_0813 mgsm_native_cot_ru_0813 mgsm_native_cot_sw_0813 mgsm_native_cot_te_0813 mgsm_native_cot_th_0813 mgsm_native_cot_bn_0813
