# spec_guard

## AgentDojo quick start (Conda `llm`)

This repository includes a runnable AgentDojo setup:

- `agentdojo/setup_agentdojo.sh`
- `agentdojo/run_smoke_check.sh`
- `agentdojo/run_openrouter_benchmark.py` (for OpenRouter custom models)
- `agentdojo/AGENTDOJO_TUTORIAL.md`

Quick run:

```bash
cd agentdojo
./setup_agentdojo.sh
./run_smoke_check.sh
```

OpenRouter benchmark example:

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -s workspace \
  -ut user_task_0 \
  --defense tool_filter
```

Custom entry example (loads custom attack/defense module):

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -ml custom_entries_example \
  --list-entries
```

## Attack Introduction

The project currently supports the following built-in attacks (via AgentDojo):

- `manual`: Human-in-the-loop manual injection content.
- `direct`: Simple direct instruction (`TODO: {goal}`).
- `ignore_previous`: Classic "ignore previous instructions" style jailbreak.
- `system_message`: Attempts to impersonate system-level instruction.
- `injecagent`: Template inspired by InjecAgent-style prompting.
- `important_instructions`: Signed "important message" attack template.
- `important_instructions_no_user_name`: Variant without explicit user identity.
- `important_instructions_no_model_name`: Variant without explicit model identity.
- `important_instructions_no_names`: Variant without user and model identity.
- `important_instructions_wrong_model_name`: Variant using incorrect model identity.
- `important_instructions_wrong_user_name`: Variant using incorrect user identity.
- `tool_knowledge`: Attack augmented with tool-sequence knowledge from injection task ground truth.
- `dos`: Denial-of-service style interruption attack.
- `swearwords_dos`: Offensive-content DoS variant.
- `captcha_dos`: Captcha/refusal-triggering DoS variant.
- `offensive_email_dos`: Harmful-email request DoS variant.
- `felony_dos`: Illegal-content warning DoS variant.

Custom attacks can be added by creating a module and using AgentDojo's `@register_attack`, then loading it with `-ml <module_name>`.

## Defense Introduction

The project currently supports the following built-in defenses:

- `tool_filter`: LLM-based filtering of tools before use.
- `transformers_pi_detector`: Prompt-injection detector model in the tool loop.
- `spotlighting_with_delimiting`: Delimits tool output and instructs model to distrust delimited content.
- `repeat_user_prompt`: Re-inserts user intent during execution loop.

The repo also supports custom defenses through `RegisterEntry`-style registration in:

- `agentdojo/custom_registry.py`

Use `@register_openrouter_defense("name")` in your custom module, then load with `-ml <module_name>`.

## Example Usage

### 1) Install and verify

```bash
cd /home/xianglin/git_space/spec_guard/agentdojo
./setup_agentdojo.sh
./run_smoke_check.sh
```

### 2) List available attacks/defenses (including custom entries)

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -ml custom_entries_example \
  --list-entries
```

### 3) Run without attack (utility-focused)

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -s workspace \
  -ut user_task_0
```

### 4) Run with attack + defense

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -s workspace \
  -ut user_task_0 \
  --attack tool_knowledge \
  --defense tool_filter
```

### 5) Run using custom registered entries

```bash
python run_openrouter_benchmark.py \
  --openrouter-model openai/gpt-4o-mini \
  -ml custom_entries_example \
  --attack prefix_policy_bypass \
  --defense block_risky_tools \
  -s workspace \
  -ut user_task_0
```

### 6) Output location and reruns

- Default logs are written to `./runs` under the current working directory.
- To force rerun existing tasks, add `-f`.
- To change output path, use `--logdir /path/to/logs`.

For full project organization and additional details, read `agentdojo/AGENTDOJO_TUTORIAL.md`.
