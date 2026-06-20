#!/usr/bin/env bash
set -uo pipefail

python3 -m scripts.collect_tuning_performance_simple \
  --requests-file prototype/two-step-autotuning/requests/gaussian_10sizes_30000.json \
  --experiment-name exp_gaussian_10sizes_30000cfg_instruction_maps \
  --collection-mode instruction-map-only \
  --workers 16 \
  --device-type cpu
