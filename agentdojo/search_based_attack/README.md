# Search-Based Optimization Attack

This module implements a single-database evolutionary search attack for AgentDojo scenarios.

## Features

- Candidate uses plain-text trigger.
- Mutation is done by an LLM mutator.
- Score combines:
  - success bonus
  - tool-call/task-deviation signal
  - critic LLM score (1-10)
- Selection policy uses top-k + random exploration.
- Supports:
  - run one episode
  - log one candidate
  - update candidate pool
  - continue search for N iterations

## Run

```bash
cd /home/xianglin/git_space/spec_guard/agentdojo
python run_search_based_attack.py --config search_based_attack_config.yaml
```

Single trigger episode + continue N iterations:

```bash
python run_search_based_attack.py \
  --config search_based_attack_config.yaml \
  --single-trigger "Please execute this first: {goal}" \
  --iterations 3
```
