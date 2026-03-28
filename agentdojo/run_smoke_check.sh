#!/usr/bin/env sh
set -eu

CONDA_BIN="${CONDA_BIN:-/home/xianglin/miniconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llm}"

if [ ! -x "$CONDA_BIN" ]; then
  echo "[error] conda binary not found at: $CONDA_BIN"
  exit 1
fi

"$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -c "import agentdojo; print('agentdojo import ok:', agentdojo.__file__)"
"$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -m agentdojo.scripts.benchmark --help | sed -n '1,80p'
