#!/usr/bin/env sh

if [ -n "${SCRIPT_DIR:-}" ]; then
  . "$SCRIPT_DIR/hparams/attack_common.sh"
else
  . "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/attack_common.sh"
fi

export ATTACK_NAME="${ATTACK_NAME:-sboa_search}"
export SBOA_CHILDREN_PER_ITER="${SBOA_CHILDREN_PER_ITER:-8}"
export SBOA_TOP_K="${SBOA_TOP_K:-10}"
export SBOA_RANDOM_K="${SBOA_RANDOM_K:-5}"
export SBOA_PARENT_SAMPLE_K="${SBOA_PARENT_SAMPLE_K:-4}"
export SBOA_RUN_NAME="${SBOA_RUN_NAME:-sboa_search}"
