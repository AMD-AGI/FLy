export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen4b='Qwen/Qwen3-4B-Thinking-2507'

qwen72b='Qwen/Qwen2.5-72B-Instruct'
qwen_code_small='Qwen/Qwen2.5-Coder-1.5B-Instruct'
qwen_code_small2='Qwen/Qwen2.5-Coder-0.5B-Instruct'
qwen7b='Qwen/Qwen2.5-7B-Instruct'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B"
qwen3_1_7b="Qwen/Qwen3-1.7B"


draft_model=${qwen_code_small2}
target_model=${qwen72b}

export HIP_VISIBLE_DEVICES='3'
para=False
export HF_ALLOW_CODE_EVAL=1

win_len=6
thre=0.3
use_ngram=True  # True False
use_rede=True

max_ngram_size=4
num_ngram_pred_tokens=6

# task=mgsm_native_cot_fr_0813
data=0821qwen05
# for win_len in 8
# do
for task in humaneval_instruct
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
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_1_rede_${task}_UseReDe${use_rede}_Usengram${use_ngram}_ngramSize${max_ngram_size}_ngramPredict${num_ngram_pred_tokens}_winL${win_len}_thre${thre}_spdtok25
done
# done

