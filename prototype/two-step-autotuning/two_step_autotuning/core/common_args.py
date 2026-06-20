from __future__ import annotations

import os
from pathlib import Path
import sys
from argparse import SUPPRESS

DEFAULT_DB = Path(
	os.environ.get(
		"TWO_STEP_AUTOTUNING_DB",
		"experiments/exp_20260415_large_scale_5000cfg/experiments.db",
	)
)


def add_common_arguments(parser) -> None:
	parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to experiment SQLite database")
	parser.add_argument("--kernel", required=True, choices=["gemm", "gaussian"])
	parser.add_argument("--input-size", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--invalid-runtime-ms", type=float, default=1.0e9)
	parser.add_argument("--seed-record-count", type=int, default=32)
	parser.add_argument("--random-seed", type=int, default=42)
	parser.add_argument(
		"--valid-config-limit",
		type=int,
		default=None,
		help="Stop after this many unique valid dataset configurations have been evaluated.",
	)


def warn_if_seed_budget_exhausts_search(args, method: str) -> None:
	if args.valid_config_limit is None:
		return
	if args.seed_record_count < args.valid_config_limit:
		return
	print(
		f"Warning: {method} has seed_record_count={args.seed_record_count} "
		f">= valid_config_limit={args.valid_config_limit}. The run can stop during "
		"seed evaluation before OpenTuner performs a real search.",
		file=sys.stderr,
	)


def add_live_arguments(parser) -> None:
	parser.add_argument("--kernel", required=True, choices=["gemm", "gaussian"])
	parser.add_argument("--input-size", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--resolver-db", default=str(DEFAULT_DB))
	parser.add_argument("--build-dir", default="build")
	parser.add_argument("--warmup-runs", type=int, default=5)
	parser.add_argument("--runs-per-config", type=int, default=11)
	parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
	parser.add_argument("--seed-record-count", type=int, default=1)
	parser.add_argument("--random-seed", type=int, default=1)
	parser.add_argument(
		"--valid-evaluation-limit",
		dest="valid_evaluation_limit",
		type=int,
		default=None,
		help="Stop after this many valid compiled-and-run kernel evaluations.",
	)
	parser.add_argument(
		"--valid-config-limit",
		dest="valid_evaluation_limit",
		type=int,
		help=SUPPRESS,
	)
	parser.add_argument("--invalid-runtime-ms", type=float, default=1.0e9)


def live_metadata(args, method: str) -> dict:
	return {
		"method": method,
		"kernel": args.kernel,
		"input_size": args.input_size,
		"output_dir": str(Path(args.output_dir)),
		"resolver_db": args.resolver_db,
		"build_dir": args.build_dir,
		"warmup_runs": args.warmup_runs,
		"runs_per_config": args.runs_per_config,
		"runtime_statistic": "median",
		"device_type": args.device_type,
		"seed_record_count": args.seed_record_count,
		"random_seed": args.random_seed,
		"valid_evaluation_limit": args.valid_evaluation_limit,
		"budget_metric": "valid_evaluation_count",
		"invalid_runtime_ms": args.invalid_runtime_ms,
		"mode": "live",
	}
