export HF_HOME=/data

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B"

draft_model=${llama31_8B}
target_model=${llama33_70B}


export HIP_VISIBLE_DEVICES='2'
export HF_ALLOW_CODE_EVAL=1
para=False

temp=0
data=0922
# task=humaneval_instruct

use_ngram=True
max_ngram_size=4
num_ngram_pred_tokens=6
thre=0.3
use_rede=True
win_len=6


for task in niah_multivalue acp_prog_gen
do
for trial in 0 1
do
    python -m lm_eval \
        --model hf \
        --model_args temperature=${temp},use_ngram=${use_ngram},max_ngram_size=${max_ngram_size},num_ngram_pred_tokens=${num_ngram_pred_tokens},entropy_thre=${thre},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=15,revise_decoding=${use_rede},win_len=${win_len} \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_llama33_${task}_temp${temp}_trial${trial}
done
done

# humaneval_instruct acp_prog_gen gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 mgsm_native_cot_th_0813 mbpp_plus_instruct niah_multivalue
