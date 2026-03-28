#!/usr/bin/env sh
set -eu

CONDA_BIN="${CONDA_BIN:-/home/xianglin/miniconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llm}"
INSTALL_TRANSFORMERS="${INSTALL_TRANSFORMERS:-0}"

if [ ! -x "$CONDA_BIN" ]; then
  echo "[error] conda binary not found at: $CONDA_BIN"
  echo "[hint] Set CONDA_BIN to your conda executable path."
  exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "[error] Conda environment '$CONDA_ENV_NAME' not found."
  echo "[hint] Existing envs:"
  "$CONDA_BIN" env list
  exit 1
fi

"$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip

if [ "$INSTALL_TRANSFORMERS" = "1" ]; then
  "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -m pip install "agentdojo[transformers]"
else
  "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -m pip install agentdojo
fi

cat <<MSG
[ok] AgentDojo is installed in conda env '$CONDA_ENV_NAME'.

Quick commands:
1) eval "\$($CONDA_BIN shell.posix hook)"
2) conda activate "$CONDA_ENV_NAME"
3) python -m agentdojo.scripts.benchmark --help
MSG
