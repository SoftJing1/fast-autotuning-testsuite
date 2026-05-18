from __future__ import annotations

from pathlib import Path
import sys

DEFAULT_DB = Path("experiments/exp_20260415_large_scale_5000cfg/experiments.db")


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
	parser.add_argument("--build-dir", default="build")
	parser.add_argument("--runs-per-config", type=int, default=3)
	parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
	parser.add_argument("--seed-record-count", type=int, default=1)
	parser.add_argument("--random-seed", type=int, default=1)
	parser.add_argument(
		"--valid-config-limit",
		type=int,
		default=None,
		help="Stop after this many unique valid compiled-and-run kernels.",
	)
	parser.add_argument("--invalid-runtime-ms", type=float, default=1.0e9)


def live_metadata(args, method: str) -> dict:
	return {
		"method": method,
		"kernel": args.kernel,
		"input_size": args.input_size,
		"output_dir": str(Path(args.output_dir)),
		"build_dir": args.build_dir,
		"runs_per_config": args.runs_per_config,
		"device_type": args.device_type,
		"seed_record_count": args.seed_record_count,
		"random_seed": args.random_seed,
		"valid_config_limit": args.valid_config_limit,
		"invalid_runtime_ms": args.invalid_runtime_ms,
		"mode": "live",
	}
