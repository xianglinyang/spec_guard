# AgentDojo Environment and Usage Guide (Conda `llm`)

This guide sets up AgentDojo in the `spec_guard` repository using your Conda `llm` environment.

## 1) What was prepared in this repo

Inside `spec_guard/agentdojo/`:

- `setup_agentdojo.sh`: installs AgentDojo into Conda env `llm`.
- `run_smoke_check.sh`: validates installation and benchmark CLI availability.
- `run_openrouter_benchmark.py`: custom benchmark runner for OpenRouter models.
- `AGENTDOJO_TUTORIAL.md`: this runbook.

## 2) Prerequisites

- Conda available at `/home/xianglin/miniconda3/bin/conda` (default in scripts).
- Existing Conda env named `llm`.
- Internet access for package installation.
- OpenRouter API key in environment (`OPENROUTER_API_KEY` recommended).

## 3) Setup in `llm`

From `spec_guard/agentdojo`:

```bash
./setup_agentdojo.sh
```

Optional (adds local detector dependencies like `torch`):

```bash
INSTALL_TRANSFORMERS=1 ./setup_agentdojo.sh
```

If your conda path or env name differs:

```bash
CONDA_BIN=/path/to/conda CONDA_ENV_NAME=myenv ./setup_agentdojo.sh
```

## 4) Verify setup

```bash
./run_smoke_check.sh
```

It checks:

- `import agentdojo` works in `llm`
- `python -m agentdojo.scripts.benchmark --help` runs

## 5) Run with OpenRouter (custom models)

Activate env:

```bash
eval "$(/home/xianglin/miniconda3/bin/conda shell.posix hook)"
conda activate llm
```

Ensure key is set (you already have this):

```bash
export OPENROUTER_API_KEY='your-key'
```

Run a small benchmark with any OpenRouter model id:

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -s workspace \
  -ut user_task_0 \
  -ut user_task_1 \
  --defense tool_filter \
  --attack tool_knowledge
```

Run broader benchmark (all suites/tasks):

```bash
python run_openrouter_benchmark.py \
  --openrouter-model anthropic/claude-3.5-sonnet
```

Useful options:

- `--base-url` (default: `https://openrouter.ai/api/v1`)
- `--api-key-env` (default: `OPENROUTER_API_KEY`)
- `--model-alias` keeps attack templates compatible (default is good)
- `--list-entries` prints available attacks/defenses after loading custom modules

Optional OpenRouter headers:

- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_X_TITLE`

## 6) How AgentDojo is organized (upstream)

- `src/agentdojo/`: core benchmark and runtime code.
- `agentdojo/scripts/benchmark.py`: main benchmark CLI module.
- `docs/`: concepts, API docs, and results pages.
- `examples/`, `notebooks/`: usage and analysis examples.
- `runs/`: benchmark outputs in the upstream project.

In `spec_guard`, we keep bootstrap/run scripts and install the package from PyPI into Conda.

## 7) Troubleshooting

- `conda binary not found`:
  - Set `CONDA_BIN=/your/path/to/conda` when running scripts.
- `Conda environment 'llm' not found`:
  - Use `CONDA_ENV_NAME=...` or create env first.
- `Missing API key` in custom runner:
  - Set `OPENROUTER_API_KEY` (or pass `--api-key-env`).
- Model/provider capability mismatch:
  - Some OpenRouter models may have limited tool-calling compatibility. Try another model id with stronger tool-calling support.

## 8) References

- GitHub: https://github.com/ethz-spylab/agentdojo
- Docs: https://agentdojo.spylab.ai/
- Paper: https://arxiv.org/abs/2406.13352

## 9) Add custom defense and attack entries

This repo now supports custom registration for both:

- attacks: use AgentDojo's `@register_attack`
- defenses: use `@register_openrouter_defense` from `custom_registry.py`

Example module is included:

- `custom_entries_example.py`
  - defense: `block_risky_tools`
  - attack: `prefix_policy_bypass`

Load and list entries:

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -ml custom_entries_example \
  --list-entries
```

Run with custom defense:

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -ml custom_entries_example \
  --defense block_risky_tools \
  -s workspace \
  -ut user_task_0
```

Run with custom attack:

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -ml custom_entries_example \
  --attack prefix_policy_bypass \
  -s workspace \
  -ut user_task_0
```
