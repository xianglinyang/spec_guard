# Search-Based Optimization Attack

This module provides four registered adaptive attacks for AgentDojo prompt-injection benchmarks:

- `pair_search`
- `tap_search`
- `autodan_search`
- `sboa_search`

`search_based_optimization` is kept as a backward-compatible alias to `sboa_search`.

## Shared behavior

- Candidate is a plain-text trigger.
- Budget is counted per `(user_task, injection_task)` pair.
- Early stop triggers on `episode.success == True`.
- Score combines:
  - success bonus
  - tool-call/task-deviation signal
  - critic LLM score (1-10)
- Logs are written under `SBOA_RUN_LOG_DIR` with:
  - `candidates.jsonl`
  - `iterations.jsonl`
  - `summary.json`

## Baseline-specific behavior

- `pair_search`: iterative best-parent refinement.
- `tap_search`: tree/BFS expansion with dual constraints: `TAP_MAX_DEPTH` and `TAP_MAX_ITERATIONS`.
- `autodan_search`: population-based evolution with elitism, roulette selection, LLM-guided crossover, and mutation.
- `sboa_search`: single-database top-k + random exploration.

## Quick run

```bash
cd /home/xianglin/git_space/spec_guard/agentdojo
ATTACK_NAME=sboa_search ./scripts/run_search_based_optimization_attack.sh
```

Per-baseline wrappers:

```bash
./scripts/run_pair_search_attack.sh
./scripts/run_tap_search_attack.sh
./scripts/run_autodan_search_attack.sh
./scripts/run_sboa_search_attack.sh
```

## Key env vars

Shared:

- `SBOA_MUTATOR_MODEL`, `SBOA_CRITIC_MODEL`
- `SBOA_MAX_ITERATIONS`, `SBOA_MAX_POOL_SIZE`, `SBOA_SEED`
- `SBOA_SUCCESS_BONUS`, `SBOA_WEIGHT_DEVIATION`, `SBOA_WEIGHT_CRITIC`

SBOA:

- `SBOA_CHILDREN_PER_ITER`, `SBOA_TOP_K`, `SBOA_RANDOM_K`, `SBOA_PARENT_SAMPLE_K`

PAIR:

- `PAIR_CHILDREN_PER_ITER`

TAP:

- `TAP_MAX_ITERATIONS`, `TAP_MAX_DEPTH`, `TAP_BRANCHING_WIDTH`

AutoDAN:

- `AUTODAN_MAX_ITERATIONS`, `AUTODAN_POPULATION_SIZE`, `AUTODAN_NUM_ELITES`
- `AUTODAN_MUTATION_RATE`, `AUTODAN_CROSSOVER_RATE`, `AUTODAN_CHILDREN_PER_GEN`
- `AUTODAN_CROSSOVER_PROMPT_TEMPLATE`
