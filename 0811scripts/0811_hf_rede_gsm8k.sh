export HF_HOME=/shareddata/jinzeli2/cache

# model_70b=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
# model_8b=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

model_70b=${llama31_70B}
model_8b=${llama31_8B}

export HIP_VISIBLE_DEVICES='6'

win_len=8
thre=0.3
use_ngram=True  # True False
use_rede=True

max_ngram_size=4
num_ngram_pred_tokens=6


for use_ngram in True False
do
for task in gsm8k_cot_llama
do
    python -m lm_eval \
        --model hf \
        --model_args use_ngram=${use_ngram},max_ngram_size=${max_ngram_size},num_ngram_pred_tokens=${num_ngram_pred_tokens},entropy_thre=${thre},pretrained=${model_8b},parallelize=False,target_model_path=${model_70b},use_sd=True,spd_k=25,revise_decoding=${use_rede},win_len=${win_len} \
        --tasks ${task} \
        --output_path ./eval_out/0811 \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        &> ./output_log/0812_1_gpunum1_HF70B_${task}_UseReDe${use_rede}_Usengram${use_ngram}_ngramSize${max_ngram_size}_ngramPredict${num_ngram_pred_tokens}_winL${win_len}
done
done

