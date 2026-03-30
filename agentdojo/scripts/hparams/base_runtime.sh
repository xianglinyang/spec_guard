#!/usr/bin/env sh

# Base runtime/model/benchmark knobs shared by all runs.

: "${OPENROUTER_API_KEY:=}"
: "${LOCAL_OPENAI_API_KEY:=EMPTY}"

export OPENROUTER_MODEL="${OPENROUTER_MODEL:-openai/gpt-4o-mini}"
export BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
export API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"
export MODEL_ALIAS="${MODEL_ALIAS:-gpt}"

export BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.2}"
export SUITE="${SUITE:-workspace}"
export USER_TASK="${USER_TASK:-user_task_0}"
export INJECTION_TASK="${INJECTION_TASK:-injection_task_0}"
export LOGDIR="${LOGDIR:-./runs}"
export FORCE_RERUN="${FORCE_RERUN:-0}"
export MODULES="${MODULES:-}"

# Keep default unset; set if needed by caller.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
