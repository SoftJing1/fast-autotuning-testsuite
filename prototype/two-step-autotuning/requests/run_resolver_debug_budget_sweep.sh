#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/resolver_debug_budget_sweep}"
RESOLVER_DB="${RESOLVER_DB:-../../experiments/exp_20260518_large_scale_30000cfg/experiments.db}"
BUILD_DIR="${BUILD_DIR:-../../build}"
VALID_LIMIT="${VALID_LIMIT:-300}"
TEST_LIMIT="${TEST_LIMIT:-40000}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-7}"
MAX_VALUES_PER_OP="${MAX_VALUES_PER_OP:-24}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"
DISTANCE_METRIC="${DISTANCE_METRIC:-normalized-euclidean}"
# CPU OpenCL timing is only authoritative when one measured experiment owns the
# device. Override JOBS for exploratory throughput runs, not final comparisons.
JOBS="${JOBS:-1}"
INCLUDE_PARAMETER_BASELINE="${INCLUDE_PARAMETER_BASELINE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROTOTYPE_DIR}/../.." && pwd)"
export PYTHONPATH="${PROTOTYPE_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"

CASES=(
  "gaussian:512x512:gaussian_512"
  "gaussian:1024x1024:gaussian_1024"
  "gaussian:2048x2048:gaussian_2048"
  "gemm:128x128x128:gemm_128"
  "gemm:256x256x256:gemm_256"
  "gemm:512x512x512:gemm_512"
)
SEEDS=(1 2 3 4 5)
INNER_VALID_PROFILE_LIMITS=(16 32 64)

RUNNING_JOBS=0

wait_for_slot() {
  while (( RUNNING_JOBS >= JOBS )); do
    wait -n
    RUNNING_JOBS=$((RUNNING_JOBS - 1))
  done
}

run_experiment() {
  wait_for_slot
  (
    echo "[start] $*"
    "$@"
    echo "[done] $*"
  ) &
  RUNNING_JOBS=$((RUNNING_JOBS + 1))
}

wait_for_all() {
  while (( RUNNING_JOBS > 0 )); do
    wait -n
    RUNNING_JOBS=$((RUNNING_JOBS - 1))
  done
}

if (( JOBS < 1 )); then
  echo "JOBS must be >= 1" >&2
  exit 2
fi

for case_spec in "${CASES[@]}"; do
  IFS=":" read -r kernel input_size case_name <<<"${case_spec}"
  for seed in "${SEEDS[@]}"; do
    if [[ "${INCLUDE_PARAMETER_BASELINE}" == "1" ]]; then
      parameter_out="${OUTPUT_ROOT}/parameter/${case_name}/seed_${seed}/parameter_tuning"
      mkdir -p "${parameter_out}"
      run_experiment python3 -m two_step_autotuning.live.parameter_tuner \
        --kernel "${kernel}" \
        --input-size "${input_size}" \
        --output-dir "${parameter_out}" \
        --database "sqlite:///${parameter_out}/opentuner.db" \
        --build-dir "${BUILD_DIR}" \
        --device-type "${DEVICE_TYPE}" \
        --random-seed "${seed}" \
        --valid-evaluation-limit "${VALID_LIMIT}" \
        --warmup-runs "${WARMUP_RUNS}" \
        --runs-per-config "${RUNS_PER_CONFIG}" \
        --test-limit "${TEST_LIMIT}"
    fi

    db_only_out="${OUTPUT_ROOT}/db_only/${case_name}/seed_${seed}/instruction_map_tuning"
    mkdir -p "${db_only_out}"
    run_experiment python3 -m two_step_autotuning.live.two_step_tuner \
      --kernel "${kernel}" \
      --input-size "${input_size}" \
      --output-dir "${db_only_out}" \
      --database "sqlite:///${db_only_out}/opentuner.db" \
      --resolver-db "${RESOLVER_DB}" \
      --build-dir "${BUILD_DIR}" \
      --device-type "${DEVICE_TYPE}" \
      --random-seed "${seed}" \
      --valid-evaluation-limit "${VALID_LIMIT}" \
      --warmup-runs "${WARMUP_RUNS}" \
      --runs-per-config "${RUNS_PER_CONFIG}" \
      --test-limit "${TEST_LIMIT}" \
      --max-values-per-op "${MAX_VALUES_PER_OP}" \
      --resolver-distance-metric "${DISTANCE_METRIC}" \
      --resolver-refinement-mode database

    for limit in "${INNER_VALID_PROFILE_LIMITS[@]}"; do
      instruction_out="${OUTPUT_ROOT}/inner_${limit}/${case_name}/seed_${seed}/instruction_map_tuning"
      mkdir -p "${instruction_out}"
      run_experiment python3 -m two_step_autotuning.live.two_step_tuner \
        --kernel "${kernel}" \
        --input-size "${input_size}" \
        --output-dir "${instruction_out}" \
        --database "sqlite:///${instruction_out}/opentuner.db" \
        --resolver-db "${RESOLVER_DB}" \
        --build-dir "${BUILD_DIR}" \
        --device-type "${DEVICE_TYPE}" \
        --random-seed "${seed}" \
        --valid-evaluation-limit "${VALID_LIMIT}" \
        --warmup-runs "${WARMUP_RUNS}" \
        --runs-per-config "${RUNS_PER_CONFIG}" \
        --test-limit "${TEST_LIMIT}" \
        --max-values-per-op "${MAX_VALUES_PER_OP}" \
        --resolver-distance-metric "${DISTANCE_METRIC}" \
        --resolver-refinement-mode inner-tuner \
        --resolver-inner-valid-profile-limit "${limit}" \
        --resolver-inner-test-limit 512 \
        --resolver-inner-target-ratio 0.1
    done
  done
done

wait_for_all

report_args=()
if [[ "${INCLUDE_PARAMETER_BASELINE}" == "1" ]]; then
  report_args+=(--experiment "parameter=${OUTPUT_ROOT}/parameter:parameter_tuning")
fi
report_args+=(
  --experiment "db_only=${OUTPUT_ROOT}/db_only"
  --experiment "inner_16=${OUTPUT_ROOT}/inner_16" \
  --experiment "inner_32=${OUTPUT_ROOT}/inner_32" \
  --experiment "inner_64=${OUTPUT_ROOT}/inner_64" \
  --output-dir "${OUTPUT_ROOT}/report" \
  --valid-limit "${VALID_LIMIT}"
)

python3 -m two_step_autotuning.visualization.compare_distance_metrics_report "${report_args[@]}"
