from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..core.common_args import DEFAULT_DB
from ..core.dataset import TuningDataset
from ..core.instruction_space import (
	INST_INDEX_PREFIX,
	build_instruction_parameter_specs,
	counts_from_tuning_indices,
	nearest_indices_from_counts,
)
from ..core.resolver import DatabaseApproxInstructionMapResolver, ResolverWeights
from .config_space import LiveConfigGenerator
from .executor import LiveKernelExecutor, check_instruction_counter_available


def _next_index_config(start_index_config: dict[str, int], specs: dict) -> dict[str, int]:
	"""Construct a deterministic non-seed instruction-map candidate."""
	next_config = {}
	for op in sorted(specs):
		key = f"{INST_INDEX_PREFIX}{op}"
		values = specs[op].values
		start_index = int(start_index_config[key])
		if len(values) <= 1:
			next_config[key] = start_index
		elif start_index < len(values) - 1:
			next_config[key] = len(values) - 1
		else:
			next_config[key] = 0
	return next_config


def _compact_profile(profile) -> dict[str, Any]:
	return {
		"exp_id": profile.exp_id,
		"param_hash": profile.param_hash,
		"config": profile.config,
		"instruction_map": dict(sorted(profile.raw_counts.items())),
		"runtime_ms": profile.runtime_ms,
	}


def run_diagnostic(args) -> dict[str, Any]:
	check_instruction_counter_available()
	generator = LiveConfigGenerator(
		args.kernel,
		args.input_size,
		device_type=args.device_type,
		random_seed=args.random_seed,
	)
	executor = LiveKernelExecutor(
		args.output_dir,
		args.kernel,
		args.input_size,
		build_dir=args.build_dir,
		runs_per_config=args.runs_per_config,
	)
	dataset = TuningDataset(Path(args.resolver_db), args.kernel, args.input_size)
	specs = build_instruction_parameter_specs(
		dataset,
		max_values_per_op=args.max_values_per_op,
	)
	if not specs:
		raise RuntimeError("resolver dataset produced no varying instruction-map parameters")
	start_result = executor.profile_config(generator.generate(1)[0])
	if not start_result.valid or start_result.profile is None:
		raise RuntimeError("diagnostic could not profile the shared start config")
	start_profile = start_result.profile

	resolver = DatabaseApproxInstructionMapResolver(
		dataset,
		weights=ResolverWeights(
			raw=args.raw_weight,
			normalized=args.normalized_weight,
			total=args.total_weight,
		),
	)

	start_index_config = nearest_indices_from_counts(start_profile.raw_counts, specs)
	next_index_config = _next_index_config(start_index_config, specs)
	default_counts = dataset.median_instruction_counts()
	requested_counts = dict(default_counts)
	requested_counts.update(counts_from_tuning_indices(next_index_config, specs))

	resolution = resolver.resolve(requested_counts)
	if not resolution.valid or resolution.record is None:
		payload = {
			"status": "resolver_invalid",
			"requested_instruction_map": requested_counts,
			"resolver_result": {
				"status": resolution.status,
				"distance": resolution.distance,
				"raw_distance": resolution.raw_distance,
				"mix_distance": resolution.mix_distance,
				"total_distance": resolution.total_distance,
				"duplicate_count": resolution.duplicate_count,
				"resolver_time_ms": resolution.resolver_time_ms,
			},
		}
	else:
		execution = executor.execute_config(resolution.record.config)
		if not execution.valid or execution.profile is None:
			raise RuntimeError(f"resolved config failed execution: {execution.error_msg}")
		payload = {
			"status": "ok",
			"kernel": args.kernel,
			"input_size": args.input_size,
			"encoding": "online_instruction_value_index",
			"distance_metric": {
				"name": "euclidean",
				"vector": "per-op raw instruction counts",
			},
			"instruction_space_source": "resolver_dataset_full",
			"representative_values_by_opcode": {
				op: list(spec.values)
				for op, spec in sorted(specs.items())
			},
			"starting_seed": {
				"index_config": start_index_config,
				"profile": _compact_profile(start_profile),
			},
			"next_instruction_map_candidate": {
				"index_config": next_index_config,
				"requested_instruction_map": dict(sorted(requested_counts.items())),
			},
			"resolved_best_close_config": {
				"profile_before_execution": _compact_profile(resolution.record),
				"executed_profile": _compact_profile(execution.profile),
				"resolver_result": {
					"status": resolution.status,
					"distance": resolution.distance,
					"raw_distance": resolution.raw_distance,
					"mix_distance": resolution.mix_distance,
					"total_distance": resolution.total_distance,
					"duplicate_count": resolution.duplicate_count,
					"resolver_time_ms": resolution.resolver_time_ms,
				},
				"runtime_ms": execution.runtime_ms,
			},
		}

	output_path = Path(args.output_dir) / "diagnostic.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
	payload["diagnostic_path"] = str(output_path)
	return payload


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Trace one concrete live two-step resolution.")
	parser.add_argument("--kernel", required=True, choices=["gemm", "gaussian"])
	parser.add_argument("--input-size", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--resolver-db", default=str(DEFAULT_DB))
	parser.add_argument("--build-dir", default="build")
	parser.add_argument("--runs-per-config", type=int, default=1)
	parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
	parser.add_argument("--random-seed", type=int, default=1)
	parser.add_argument("--max-values-per-op", type=int, default=12)
	parser.add_argument("--raw-weight", type=float, default=0.4)
	parser.add_argument("--normalized-weight", type=float, default=0.5)
	parser.add_argument("--total-weight", type=float, default=0.1)
	return parser


def main() -> int:
	args = build_argparser().parse_args()
	payload = run_diagnostic(args)
	print(json.dumps(payload, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
