export HF_HOME=/shareddata/jinzeli2/cache

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_70B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='5'
para=False



for use_ngram in True
do
for task in mgsm_native_cot_de mgsm_native_cot_es mgsm_native_cot_fr mgsm_native_cot_ru mgsm_native_cot_th mgsm_native_cot_zh
do
    python -m lm_eval \
        --model hf \
        --model_args pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/0812_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        &> ./output_log/0812_2_gpunum1_HF_baseline70B_${task}
done
done

