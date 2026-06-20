from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from ..core.common_args import DEFAULT_DB
from ..core.dataset import TuningDataset
from ..core.instruction_space import build_instruction_parameter_specs
from ..core.resolver import DatabaseApproxInstructionMapResolver, INSTRUCTION_DISTANCE_METRICS
from ..core.types import LiveProfile
from .config_space import LiveConfigGenerator
from .executor import LiveKernelExecutor, check_instruction_counter_available
from .inner_parameter_tuner import tune_inner_parameter_config


FIELDS = [
	"request_id",
	"status",
	"db_distance",
	"final_distance",
	"distance_ratio",
	"target_distance",
	"target_hit",
	"inner_valid_profile_count",
	"inner_total_tests",
	"inner_invalid_count",
	"inner_invalid_rate",
	"inner_wall_time_ms",
	"db_param_hash",
	"chosen_param_hash",
]


def _sample_requested_counts(dataset: TuningDataset, specs: dict[str, Any], rng: random.Random) -> dict[str, int]:
	counts = dict(dataset.median_instruction_counts())
	for op, spec in specs.items():
		counts[op] = int(rng.choice(spec.values))
	return counts


def _percentile(values: list[float], percentile: float) -> float | None:
	if not values:
		return None
	ordered = sorted(values)
	index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
	return ordered[index]


def run_diagnostic(args) -> dict[str, Any]:
	check_instruction_counter_available()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	dataset = TuningDataset(Path(args.resolver_db), args.kernel, args.input_size, require_runtime=False)
	specs = build_instruction_parameter_specs(dataset, max_values_per_op=args.max_values_per_op)
	if not specs:
		raise RuntimeError("resolver dataset produced no varying instruction-map parameters")

	generator = LiveConfigGenerator(
		args.kernel,
		args.input_size,
		device_type=args.device_type,
		random_seed=args.random_seed,
	)
	executor = LiveKernelExecutor(
		output_dir,
		args.kernel,
		args.input_size,
		build_dir=args.build_dir,
		warmup_runs=0,
		runs_per_config=1,
	)
	resolver = DatabaseApproxInstructionMapResolver(
		dataset,
		metric=args.resolver_distance_metric,
	)
	rng = random.Random(args.random_seed)
	rows = []
	for request_id in range(1, args.request_count + 1):
		requested_counts = _sample_requested_counts(dataset, specs, rng)
		resolution = resolver.resolve(requested_counts)
		if not resolution.valid or resolution.record is None:
			rows.append(
				{
					"request_id": request_id,
					"status": resolution.status,
					"db_distance": resolution.distance,
				}
			)
			continue
		db_profile = LiveProfile(
			exp_id=resolution.record.exp_id,
			kernel_type=resolution.record.kernel_type,
			input_size=resolution.record.input_size,
			param_hash=resolution.record.param_hash,
			config=resolution.record.config,
			raw_counts=resolution.record.raw_counts,
		)
		result = tune_inner_parameter_config(
			args,
			generator,
			executor,
			requested_counts,
			db_profile,
			resolution.distance,
			normalization_scales=resolver.normalization_scales
			if args.resolver_distance_metric == "normalized-euclidean"
			else None,
		)
		invalid_rate = result.invalid_count / result.total_tests if result.total_tests else 0.0
		rows.append(
			{
				"request_id": request_id,
				"status": result.source,
				"db_distance": resolution.distance,
				"final_distance": result.distance,
				"distance_ratio": result.distance_ratio,
				"target_distance": result.target_distance,
				"target_hit": result.target_hit,
				"inner_valid_profile_count": result.valid_profile_count,
				"inner_total_tests": result.total_tests,
				"inner_invalid_count": result.invalid_count,
				"inner_invalid_rate": invalid_rate,
				"inner_wall_time_ms": result.wall_time_ms,
				"db_param_hash": db_profile.param_hash,
				"chosen_param_hash": result.profile.param_hash,
			}
		)

	csv_path = output_dir / "inner_parameter_tuner_diagnostic.csv"
	with csv_path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)

	ratios = [
		float(row["distance_ratio"])
		for row in rows
		if row.get("distance_ratio") is not None
	]
	target_hits = [bool(row.get("target_hit")) for row in rows if "target_hit" in row]
	summary = {
		"kernel": args.kernel,
		"input_size": args.input_size,
		"request_count": args.request_count,
		"distance_metric": args.resolver_distance_metric,
		"inner_valid_profile_limit": args.resolver_inner_valid_profile_limit,
		"inner_test_limit": args.resolver_inner_test_limit,
		"target_ratio": args.resolver_inner_target_ratio,
		"median_distance_ratio": statistics.median(ratios) if ratios else None,
		"best_distance_ratio": min(ratios) if ratios else None,
		"p90_distance_ratio": _percentile(ratios, 0.9),
		"target_hit_rate": (sum(target_hits) / len(target_hits)) if target_hits else None,
		"rows": rows,
		"csv_path": str(csv_path),
	}
	json_path = output_dir / "inner_parameter_tuner_diagnostic.json"
	json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
	summary["json_path"] = str(json_path)
	return summary


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Diagnose nested inner parameter tuning distance reduction.")
	parser.add_argument("--kernel", required=True, choices=["gemm", "gaussian"])
	parser.add_argument("--input-size", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--resolver-db", default=str(DEFAULT_DB))
	parser.add_argument("--build-dir", default="build")
	parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
	parser.add_argument("--random-seed", type=int, default=1)
	parser.add_argument("--max-values-per-op", type=int, default=12)
	parser.add_argument("--request-count", type=int, default=5)
	parser.add_argument(
		"--resolver-distance-metric",
		choices=INSTRUCTION_DISTANCE_METRICS,
		default="normalized-euclidean",
	)
	parser.add_argument("--resolver-inner-valid-profile-limit", type=int, default=32)
	parser.add_argument("--resolver-inner-test-limit", type=int, default=512)
	parser.add_argument("--resolver-inner-target-ratio", type=float, default=0.1)
	parser.add_argument("--resolver-inner-invalid-distance-penalty", type=float, default=1.0e12)
	return parser


def main() -> int:
	args = build_argparser().parse_args()
	payload = run_diagnostic(args)
	print(json.dumps(payload, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
