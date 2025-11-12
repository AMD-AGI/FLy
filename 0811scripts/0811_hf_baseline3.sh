export HF_HOME=/shareddata/jinzeli2/cache

# target_model=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
# draft_model=/shareddata/jinzeli2/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

# target_model=${llama31_70B}
draft_model=${llama31_8B}

# export HIP_VISIBLE_DEVICES='4,5,6,7'
export HIP_VISIBLE_DEVICES='7'
export HF_ALLOW_CODE_EVAL=1


# win_len=8
# thre=0.3
# use_ngram=True  # True False
# use_rede=True

# max_ngram_size=4
# num_ngram_pred_tokens=6

target_model_list=(llama31_70B)

for target_model in "${target_model_list[@]}"
do
for task in gsm8k_cot_llama
do
    t_model_path=${!target_model}
    python -m lm_eval \
        --model hf \
        --model_args pretrained=${draft_model},parallelize=False,target_model_path=${t_model_path},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/0811 \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/0811_2_gpunum1_HF_${task}_${target_model}
done
done
