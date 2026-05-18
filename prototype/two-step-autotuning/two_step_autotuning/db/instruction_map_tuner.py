from __future__ import annotations

import argparse
import json

import opentuner
from opentuner import MeasurementInterface, Result

from ..core.common_args import add_common_arguments, warn_if_seed_budget_exhausts_search
from ..core.dataset import TuningDataset
from ..core.instruction_space import (
	build_instruction_manipulator,
	build_instruction_parameter_specs,
	counts_from_tuning_indices,
	indices_from_counts,
)
from ..core.resolver import DatabaseApproxInstructionMapResolver, ResolverWeights
from ..core.result_recorder import ResultRecorder


class InstructionMapTuningInterface(MeasurementInterface):
	def __init__(self, args):
		super().__init__(args)
		warn_if_seed_budget_exhausts_search(args, "instruction_map_tuning")
		self.dataset = TuningDataset(args.db, args.kernel, args.input_size)
		self.parameter_specs = build_instruction_parameter_specs(
			self.dataset,
			max_values_per_op=args.max_values_per_op,
			top_config_count=args.top_config_count,
			min_log_range=args.min_log_range,
			min_log_ratio=args.min_log_ratio,
		)
		self.default_counts = self.dataset.median_instruction_counts()
		self._manipulator = build_instruction_manipulator(self.parameter_specs)
		self.resolver = DatabaseApproxInstructionMapResolver(
			self.dataset,
			weights=ResolverWeights(
				raw=args.raw_weight,
				normalized=args.normalized_weight,
				total=args.total_weight,
			),
			max_distance=args.max_resolver_distance,
		)
		self.recorder = ResultRecorder(
			args.output_dir,
			{
				"method": "instruction_map_tuning",
				"db": args.db,
				"kernel": args.kernel,
				"input_size": args.input_size,
				"invalid_runtime_ms": args.invalid_runtime_ms,
				"record_count": len(self.dataset.records),
				"seed_record_count": args.seed_record_count,
				"random_seed": args.random_seed,
				"valid_config_limit": args.valid_config_limit,
				"opcodes": sorted(self.parameter_specs),
				"encoding": "observed_value_index",
				"instruction_value_counts": {
					op: len(spec.values)
					for op, spec in sorted(self.parameter_specs.items())
				},
				"raw_weight": args.raw_weight,
				"normalized_weight": args.normalized_weight,
				"total_weight": args.total_weight,
				"max_resolver_distance": args.max_resolver_distance,
			},
		)

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		seeds = []
		for record in self.dataset.sampled_records(self.args.seed_record_count, self.args.random_seed):
			seeds.append(indices_from_counts(record.raw_counts, self.parameter_specs))
		return seeds

	def run(self, desired_result, input, limit):
		counts = dict(self.default_counts)
		counts.update(counts_from_tuning_indices(desired_result.configuration.data, self.parameter_specs))
		result = self.resolver.resolve(counts)
		if not result.valid or result.record is None:
			self.recorder.record(
				{
					"candidate_status": result.status,
					"runtime_ms": self.args.invalid_runtime_ms,
					"resolver_distance": result.distance,
					"raw_distance": result.raw_distance,
					"mix_distance": result.mix_distance,
					"total_distance": result.total_distance,
					"resolver_time_ms": result.resolver_time_ms,
					"duplicate_count": result.duplicate_count,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)
		record = result.record
		self.recorder.record(
			{
				"candidate_status": result.status,
				"runtime_ms": record.runtime_ms,
				"exp_id": record.exp_id,
				"param_hash": record.param_hash,
				"resolver_distance": result.distance,
				"raw_distance": result.raw_distance,
				"mix_distance": result.mix_distance,
				"total_distance": result.total_distance,
				"resolver_time_ms": result.resolver_time_ms,
				"duplicate_count": result.duplicate_count,
			}
		)
		return Result(time=record.runtime_ms)

	def extra_convergence_criteria(self, result):
		limit = self.args.valid_config_limit
		return limit is not None and len(self.recorder.seen_exp_ids) >= limit

	def save_final_config(self, config):
		index_config = dict(config.data)
		requested_counts = dict(self.default_counts)
		requested_counts.update(counts_from_tuning_indices(index_config, self.parameter_specs))
		result = self.resolver.resolve(requested_counts)
		payload = {
			"encoding": "observed_value_index",
			"index_config": index_config,
			"requested_instruction_counts": requested_counts,
			"resolver_result": {
				"status": result.status,
				"distance": result.distance,
				"raw_distance": result.raw_distance,
				"mix_distance": result.mix_distance,
				"total_distance": result.total_distance,
				"duplicate_count": result.duplicate_count,
			},
			"resolved_config": None if result.record is None else result.record.config,
			"db_match": None
			if result.record is None
			else {
				"exp_id": result.record.exp_id,
				"param_hash": result.record.param_hash,
				"runtime_ms": result.record.runtime_ms,
			},
		}
		(self.recorder.output_dir / "final_config.json").write_text(
			json.dumps(payload, indent=2, sort_keys=True)
		)
		self.recorder.write_summary()


def build_argparser():
	parser = opentuner.default_argparser()
	add_common_arguments(parser)
	parser.add_argument("--max-values-per-op", type=int, default=12)
	parser.add_argument("--top-config-count", type=int, default=32)
	parser.add_argument("--min-log-range", type=int, default=1024)
	parser.add_argument("--min-log-ratio", type=float, default=16.0)
	parser.add_argument("--raw-weight", type=float, default=0.4)
	parser.add_argument("--normalized-weight", type=float, default=0.5)
	parser.add_argument("--total-weight", type=float, default=0.1)
	parser.add_argument("--max-resolver-distance", type=float, default=None)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	InstructionMapTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
