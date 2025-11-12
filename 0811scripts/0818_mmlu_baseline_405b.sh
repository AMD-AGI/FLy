# export HF_HOME=/workspace/cache
export HF_HOME=/data


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
mistral_large="mistralai/Mistral-Large-Instruct-2411"

target_model=${llama31_405B}
draft_model=${llama31_8B}

# export HIP_VISIBLE_DEVICES='4,5,6,7'
export HIP_VISIBLE_DEVICES='1,2,3'
# export HIP_VISIBLE_DEVICES='5,6,7'
# export HIP_VISIBLE_DEVICES='2,3,4'
# export HIP_VISIBLE_DEVICES='1,2,3'
# export HIP_VISIBLE_DEVICES='1,2'
# export HIP_VISIBLE_DEVICES='5'
para=True

data=0818


for lang in zh
do
for subject in computer_science chemistry business math law biology economics engineering history health other philosophy physics psychology
# for subject in math
do
    # task=mmlu_prox_${lang}_${subject}
    task=mmlu_prox_${lang}
    python -m lm_eval \
        --model hf \
        --model_args pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        &> ./output_log/${data}_test_baseline405b_${task}
done
done

# de fr es th zh en
# computer_science chemistry business math engineering physics
