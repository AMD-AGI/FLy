## Installation

```bash
git clone <repo-url>
cd FLy
pip install -e .
```

## Usage

```bash
fly --model hf \
    --model_args pretrained=<draft_model_path>,config_path=fly_config/FLy_Llama3_70b.json \
    --tasks humaneval \
    --batch_size 1
```

### Key Arguments

- `--model hf` — HuggingFace model backend
- `--model_args` — `pretrained` for the draft model path, `config_path` for the FLy config JSON
- `--tasks` — evaluation task (e.g. `humaneval`)
- `--batch_size` — batch size
- `--output_path` — directory or file to save results
- `--log_samples` — log per-sample model outputs

### List Available Tasks

```bash
fly --tasks list
```

## Configuration

Edit the JSON config in `fly_config/` to control speculative decoding behavior. See `fly_config/FLy_Llama3_70b.json` for an example.

## License

MIT. See [LICENSE.md](LICENSE.md).
