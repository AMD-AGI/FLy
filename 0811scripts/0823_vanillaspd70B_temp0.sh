export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

target_model=${llama31_70B}
draft_model=${llama31_8B}

export HIP_VISIBLE_DEVICES='5'
export HF_ALLOW_CODE_EVAL=1
para=False

use_rede=False
win_len=15
temp=0

data=0823VanillaSpd

for task in humaneval_instruct mbpp_plus_instruct
do
    python -m lm_eval \
        --model hf \
        --model_args temperature=${temp},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=${win_len},revise_decoding=${use_rede},win_len=${win_len} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        --limit 80 \
        &> ./output_log/${data}_70B_${task}_winl${win_len}_temp${temp}
done
