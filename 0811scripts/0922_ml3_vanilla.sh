export HF_HOME=/data

llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'
qwen30b='Qwen/Qwen3-30B-A3B-Thinking-2507'
qwen235b='Qwen/Qwen3-235B-A22B-Thinking-2507'
qwen3_32b="Qwen/Qwen3-32B"
qwen3_4b="Qwen/Qwen3-4B"
ml3="meta-llama/Meta-Llama-3-70B-Instruct"
ml3_8b="meta-llama/Meta-Llama-3-8B-Instruct"

draft_model=${ml3_8b}
target_model=${ml3_8b}


export HIP_VISIBLE_DEVICES='1'
export HF_ALLOW_CODE_EVAL=1
para=False

temp=0
data=0923
# task=humaneval_instruct

use_ngram=True
max_ngram_size=4
num_ngram_pred_tokens=6
thre=0.3
use_rede=True
win_len=6


for task in humaneval_instruct gsm8k_cot_llama mbpp_plus_instruct
do
for trial in 0
do
    python -m lm_eval \
        --model hf \
        --model_args temperature=${temp},pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
        --tasks ${task} \
        --output_path ./eval_out/${data}_${task} \
        --show_config \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --confirm_run_unsafe_code \
        &> ./output_log/${data}_ml3_${task}_temp${temp}_trial${trial}_vanilla_8b
done
done

# humaneval_instruct acp_prog_gen gsm8k_cot_llama mgsm_native_cot_fr_0813 mgsm_native_cot_de_0813 mgsm_native_cot_th_0813 mbpp_plus_instruct niah_multivalue
