#!/usr/bin/env sh

if [ -n "${SCRIPT_DIR:-}" ]; then
  . "$SCRIPT_DIR/hparams/attack_common.sh"
else
  . "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/attack_common.sh"
fi

export ATTACK_NAME="${ATTACK_NAME:-tap_search}"
export TAP_MAX_ITERATIONS="${TAP_MAX_ITERATIONS:-5}"
export TAP_MAX_DEPTH="${TAP_MAX_DEPTH:-3}"
export TAP_BRANCHING_WIDTH="${TAP_BRANCHING_WIDTH:-3}"
export SBOA_RUN_NAME="${SBOA_RUN_NAME:-tap_search}"
