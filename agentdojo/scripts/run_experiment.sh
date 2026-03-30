#!/usr/bin/env bash
set -euo pipefail

# Unified entrypoint for selecting attack + defense hyperparameter profiles.
#
# Usage examples:
#   ATTACK_PROFILE=sboa DEFENSE_PROFILE=spec_smoothing ./scripts/run_experiment.sh
#   ATTACK_PROFILE=tap  DEFENSE_PROFILE=none           ./scripts/run_experiment.sh
#   ATTACK_PROFILE=none DEFENSE_PROFILE=tool_filter    ./scripts/run_experiment.sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

ATTACK_PROFILE="${ATTACK_PROFILE:-sboa}"               # none|sboa|pair|tap|autodan
DEFENSE_PROFILE="${DEFENSE_PROFILE:-spec_smoothing}"   # none|tool_filter|transformers_pi_detector|repeat_user_prompt|spotlighting_with_delimiting|spec_smoothing

# Load base runtime knobs first.
. "$SCRIPT_DIR/hparams/base_runtime.sh"

# Load attack profile.
case "$ATTACK_PROFILE" in
  none) ATTACK_NAME="" ;;
  sboa) . "$SCRIPT_DIR/hparams/attack_sboa.sh" ;;
  pair) . "$SCRIPT_DIR/hparams/attack_pair.sh" ;;
  tap) . "$SCRIPT_DIR/hparams/attack_tap.sh" ;;
  autodan) . "$SCRIPT_DIR/hparams/attack_autodan.sh" ;;
  *)
    echo "[error] Unknown ATTACK_PROFILE=$ATTACK_PROFILE (supported: none|sboa|pair|tap|autodan)."
    exit 1
    ;;
esac

# Load defense profile.
DEFENSE_NAME="${DEFENSE_NAME:-}"
case "$DEFENSE_PROFILE" in
  none) DEFENSE_NAME="" ;;
  tool_filter|transformers_pi_detector|repeat_user_prompt|spotlighting_with_delimiting) DEFENSE_NAME="$DEFENSE_PROFILE" ;;
  spec_smoothing) . "$SCRIPT_DIR/hparams/defense_speculative_smoothing.sh"; DEFENSE_NAME="speculative_smoothing" ;;
  *)
    echo "[error] Unknown DEFENSE_PROFILE=$DEFENSE_PROFILE."
    exit 1
    ;;
esac

if [ -z "${OPENROUTER_API_KEY}" ]; then
  echo "[error] OPENROUTER_API_KEY is empty."
  exit 1
fi

# Preflight: local draft endpoint health check when draft uses independent local client.
if [ "${DEFENSE_PROFILE}" = "spec_smoothing" ] && [ "${SPEC_SMOOTHING_DRAFT_USE_MAIN_CLIENT:-1}" = "0" ]; then
  if ! python3 - "$SPEC_SMOOTHING_DRAFT_BASE_URL" <<'PY'
import sys
import urllib.request

base = (sys.argv[1] or "").rstrip("/")
if not base:
    print("[error] SPEC_SMOOTHING_DRAFT_BASE_URL is empty while SPEC_SMOOTHING_DRAFT_USE_MAIN_CLIENT=0.")
    raise SystemExit(1)

url = f"{base}/models"
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        code = getattr(resp, "status", 200)
        if code < 200 or code >= 500:
            print(f"[error] Draft endpoint unhealthy: {url} (status={code})")
            raise SystemExit(1)
except Exception as e:
    print(f"[error] Cannot connect to local draft endpoint: {url}")
    print(f"[error] {e}")
    raise SystemExit(1)
PY
  then
    echo "[hint] Start local OpenAI-compatible draft endpoint or set SPEC_SMOOTHING_DRAFT_USE_MAIN_CLIENT=1."
    exit 1
  fi
fi

cd "$PROJECT_DIR"

CMD="python3 run_openrouter_benchmark.py \
  --openrouter-model $OPENROUTER_MODEL \
  --base-url $BASE_URL \
  --api-key-env $API_KEY_ENV \
  --model-alias $MODEL_ALIAS \
  --benchmark-version $BENCHMARK_VERSION \
  --logdir $LOGDIR \
  -s $SUITE"

if [ -n "$ATTACK_NAME" ]; then
  CMD="$CMD --attack $ATTACK_NAME"
fi

if [ -n "${USER_TASKS:-}" ]; then
  OLD_IFS=$IFS
  IFS=','
  for ut in $USER_TASKS; do
    CMD="$CMD -ut $ut"
  done
  IFS=$OLD_IFS
else
  CMD="$CMD -ut $USER_TASK"
fi

if [ -n "$INJECTION_TASK" ]; then
  CMD="$CMD -it $INJECTION_TASK"
fi

if [ -n "$DEFENSE_NAME" ]; then
  CMD="$CMD --defense $DEFENSE_NAME"
fi

if [ "$FORCE_RERUN" = "1" ]; then
  CMD="$CMD -f"
fi

if [ -n "$MODULES" ]; then
  OLD_IFS=$IFS
  IFS=','
  for mod in $MODULES; do
    CMD="$CMD -ml $mod"
  done
  IFS=$OLD_IFS
fi

echo "[info] ATTACK_PROFILE=$ATTACK_PROFILE DEFENSE_PROFILE=$DEFENSE_PROFILE"
echo "[info] Running command:"
echo "$CMD"
# shellcheck disable=SC2086
eval $CMD
