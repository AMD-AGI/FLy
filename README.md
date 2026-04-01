# FLy: Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match (ICML 2026)

FLy is a Training-Free Loosely Speculative Decoding library for evaluation workloads.

This project is based on [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and extends the Hugging Face backend with FLy speculative decoding logic while keeping the familiar task and CLI workflow.

## Requirements

- Python 3.9+
- PyTorch 2.1+
- transformers 4.45+
- Access to the draft model and target model on Hugging Face when using Hub model IDs
- Enough GPU memory for both the draft model and the target model configured in `fly_config/`

## Installation

```bash
git clone https://github.com/AMD-AGI/FLy.git
cd FLy
pip install -e .
```

## Usage

Please reach to `start.sh` for the full example, or using the command below:
```bash
export HF_ALLOW_CODE_EVAL=1

fly --model hf \
    --model_args pretrained=<draft_model_path_or_hf_model_id>,config_path=fly_config/FLy_Llama3_70b.json \
    --tasks humaneval_instruct \
    --batch_size 1 \
    --apply_chat_template \
    --confirm_run_unsafe_code
```

`humaneval` executes reference test code during evaluation, so `HF_ALLOW_CODE_EVAL=1` and `--confirm_run_unsafe_code` are required.

If you use an instruct/chat model, keep `--apply_chat_template` enabled so prompts match the tokenizer's chat format.

The example config `fly_config/FLy_Llama3_70b.json` loads both a draft model and a target model. Make sure your machine has enough memory and that you have permission to access the referenced Hugging Face models.

## Key Arguments

- `--model hf`: Hugging Face model backend
- `--model_args`: `pretrained` for the draft model and `config_path` for the FLy JSON config
- `--tasks`: evaluation task, for example `humaneval`
- `--batch_size`: batch size; `1` is the recommended starting point for FLy runs
- `--apply_chat_template`: recommended for instruct/chat checkpoints
- `--confirm_run_unsafe_code`: required for tasks that execute code, such as `humaneval`
- `--output_path`: directory or file used to save results
- `--log_samples`: log per-sample model outputs


## Configuration

Edit the JSON config in `fly_config/` to control speculative decoding behavior. See `fly_config/FLy_Llama3_70b.json` for an example.

## Citation

```
@article{li2025training,
  title={Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match},
  author={Li, Jinze and Xu, Yixing and Li, Guanchen and Yang, Shuo and Xu, Jinfeng and Yin, Xuanwu and Li, Dong and Ngai, Edith CH and Barsoum, Emad},
  journal={arXiv preprint arXiv:2511.22972},
  year={2025}
}
```
