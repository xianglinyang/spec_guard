#!/usr/bin/env sh

if [ -n "${SCRIPT_DIR:-}" ]; then
  . "$SCRIPT_DIR/hparams/attack_common.sh"
else
  . "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/attack_common.sh"
fi

export ATTACK_NAME="${ATTACK_NAME:-pair_search}"
export PAIR_CHILDREN_PER_ITER="${PAIR_CHILDREN_PER_ITER:-1}"
export SBOA_RUN_NAME="${SBOA_RUN_NAME:-pair_search}"
