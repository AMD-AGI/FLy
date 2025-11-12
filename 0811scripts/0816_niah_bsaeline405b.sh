# -------------------------------------------------------------------------
export HF_HOME=/workspace/cache


llama33_70B='meta-llama/Llama-3.3-70B-Instruct'
llama31_70B='meta-llama/Llama-3.1-70B-Instruct'
llama31_405B='meta-llama/Llama-3.1-405B-Instruct'
llama31_8B='meta-llama/Llama-3.1-8B-Instruct'

# export HIP_VISIBLE_DEVICES='0,1,2,3'
export HIP_VISIBLE_DEVICES='4,5,6,7'
para=True
data=0816
export HF_ALLOW_CODE_EVAL=1



draft_model='meta-llama/Llama-3.1-8B-Instruct'
target_model='meta-llama/Llama-3.1-405B-Instruct'

task=niah_multivalue
#  acp_prog_gen minerva_math_algebra niah_multikey_1

python -m lm_eval \
    --model hf \
    --model_args pretrained=${draft_model},parallelize=${para},target_model_path=${target_model},use_sd=True,spd_k=0 \
    --tasks ${task} \
    --output_path ./eval_out/${data}_${task}_testrede \
    --show_config \
    --log_samples \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --confirm_run_unsafe_code \
    --trust_remote_code \
    &> ./output_log/${data}_3_405B_${task}_baseline_20value.txt
