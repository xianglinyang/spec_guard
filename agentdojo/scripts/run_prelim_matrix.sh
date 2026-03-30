#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

# Fixed prelim scope (can still be overridden if needed).
PRELIM_SUITE="${PRELIM_SUITE:-workspace}"
PRELIM_BENCHMARK_VERSION="${PRELIM_BENCHMARK_VERSION:-v1.2.2}"
PRELIM_SEEDS="${PRELIM_SEEDS:-42,43,44}"
PRELIM_DEFENSES="${PRELIM_DEFENSES:-none,tool_filter,transformers_pi_detector,repeat_user_prompt,spotlighting_with_delimiting,spec_smoothing}"
PRELIM_ATTACK_MODES="${PRELIM_ATTACK_MODES:-none,sboa}"
PRELIM_USER_TASKS="${PRELIM_USER_TASKS:-user_task_0,user_task_1,user_task_2,user_task_3,user_task_4,user_task_5,user_task_6,user_task_7,user_task_8,user_task_9}"
PRELIM_OUT_DIR="${PRELIM_OUT_DIR:-./runs_prelim}"
PRELIM_ID="${PRELIM_ID:-$(date +%Y%m%d_%H%M%S)}"
PRELIM_FORCE_RERUN="${PRELIM_FORCE_RERUN:-1}"

RUN_ROOT="$PRELIM_OUT_DIR/$PRELIM_ID"
MANIFEST="$RUN_ROOT/manifest.tsv"
mkdir -p "$RUN_ROOT"
printf "seed\tdefense\tattack_mode\tlogdir\trun_log\tgpu_log\tstatus\n" > "$MANIFEST"

start_gpu_monitor() {
  gpu_log_file="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while :; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$gpu_log_file" 2>/dev/null || true
        sleep 1
      done
    ) &
    echo "$!"
  else
    echo ""
  fi
}

OLD_IFS=$IFS
IFS=','
for seed in $PRELIM_SEEDS; do
  for defense in $PRELIM_DEFENSES; do
    for attack_mode in $PRELIM_ATTACK_MODES; do
      combo_dir="$RUN_ROOT/seed_${seed}/${defense}/${attack_mode}"
      combo_logdir="$combo_dir/runs"
      run_log="$combo_dir/run.log"
      gpu_log="$combo_dir/gpu.csv"
      mkdir -p "$combo_dir"

      gpu_pid="$(start_gpu_monitor "$gpu_log")"

      set +e
      ATTACK_PROFILE="$attack_mode" \
      DEFENSE_PROFILE="$defense" \
      SUITE="$PRELIM_SUITE" \
      BENCHMARK_VERSION="$PRELIM_BENCHMARK_VERSION" \
      USER_TASKS="$PRELIM_USER_TASKS" \
      INJECTION_TASK="" \
      LOGDIR="$combo_logdir" \
      FORCE_RERUN="$PRELIM_FORCE_RERUN" \
      SBOA_SEED="$seed" \
      SPEC_SMOOTHING_SEED="$seed" \
      "$SCRIPT_DIR/run_experiment.sh" >"$run_log" 2>&1
      status=$?
      set -e

      if [ -n "$gpu_pid" ]; then
        kill "$gpu_pid" >/dev/null 2>&1 || true
      fi

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$seed" "$defense" "$attack_mode" "$combo_logdir" "$run_log" "$gpu_log" "$status" >> "$MANIFEST"

      if [ "$status" -ne 0 ]; then
        echo "[warn] failed seed=$seed defense=$defense attack_mode=$attack_mode (see $run_log)"
      else
        echo "[ok] seed=$seed defense=$defense attack_mode=$attack_mode"
      fi
    done
  done
done
IFS=$OLD_IFS

echo "[info] prelim matrix completed: $RUN_ROOT"
echo "[info] manifest: $MANIFEST"
