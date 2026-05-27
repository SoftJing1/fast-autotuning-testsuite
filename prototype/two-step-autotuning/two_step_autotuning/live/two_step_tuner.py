from __future__ import annotations

import json
from pathlib import Path

import opentuner
from opentuner import MeasurementInterface, Result

from ..core.common_args import add_live_arguments, live_metadata
from ..core.dataset import TuningDataset
from ..core.instruction_space import (
	build_instruction_manipulator,
	build_instruction_parameter_specs,
	counts_from_tuning_indices,
	nearest_indices_from_counts,
)
from ..core.resolver import DatabaseApproxInstructionMapResolver, LiveResolutionResult, ResolverWeights
from ..core.result_recorder import ResultRecorder
from ..core.types import LiveProfile
from .config_space import LiveConfigGenerator, hash_config
from .executor import LiveKernelExecutor, check_instruction_counter_available
from .mutation import LocalInstructionMapMutator


class _DatabaseBackedLiveResolver:
	def __init__(
		self,
		dataset: TuningDataset,
		weights: ResolverWeights,
		max_distance: float | None = None,
	):
		self.dataset = dataset
		self.resolver = DatabaseApproxInstructionMapResolver(
			dataset,
			weights=weights,
			max_distance=max_distance,
		)

	def resolve(self, requested_counts: dict[str, int]):
		result = self.resolver.resolve(requested_counts)
		if not result.valid or result.record is None:
			return LiveResolutionResult(
				status=result.status,
				profile=None,
				distance=result.distance,
				raw_distance=result.raw_distance,
				mix_distance=result.mix_distance,
				total_distance=result.total_distance,
				duplicate_count=result.duplicate_count,
				resolver_time_ms=result.resolver_time_ms,
			)
		record = result.record
		return LiveResolutionResult(
			status=result.status,
			profile=LiveProfile(
				exp_id=record.exp_id,
				kernel_type=record.kernel_type,
				input_size=record.input_size,
				param_hash=record.param_hash,
				config=record.config,
				raw_counts=record.raw_counts,
			),
			distance=result.distance,
			raw_distance=result.raw_distance,
			mix_distance=result.mix_distance,
			total_distance=result.total_distance,
			duplicate_count=result.duplicate_count,
			resolver_time_ms=result.resolver_time_ms,
		)


class LiveTwoStepTuningInterface(MeasurementInterface):
	def __init__(self, args):
		super().__init__(args)
		check_instruction_counter_available()
		self.config_generator = LiveConfigGenerator(
			args.kernel,
			args.input_size,
			device_type=args.device_type,
			random_seed=args.random_seed,
		)
		self.start_config = self.config_generator.generate(1)[0]
		self.executor = LiveKernelExecutor(
			args.output_dir,
			args.kernel,
			args.input_size,
			build_dir=args.build_dir,
			warmup_runs=args.warmup_runs,
			runs_per_config=args.runs_per_config,
		)
		self.resolver_dataset = TuningDataset(
			Path(args.resolver_db),
			args.kernel,
			args.input_size,
		)
		self.parameter_specs = build_instruction_parameter_specs(
			self.resolver_dataset,
			max_values_per_op=args.max_values_per_op,
		)
		if not self.parameter_specs:
			raise RuntimeError("resolver dataset produced no varying instruction-map parameters")
		self.default_counts = self.resolver_dataset.median_instruction_counts()
		start_profile_result = self.executor.profile_config(self.start_config)
		if not start_profile_result.valid or start_profile_result.profile is None:
			raise RuntimeError("failed to profile shared start config for instruction-map seeding")
		self.start_profile = start_profile_result.profile
		self._manipulator = build_instruction_manipulator(self.parameter_specs)
		self.resolver = _DatabaseBackedLiveResolver(
			dataset=self.resolver_dataset,
			weights=ResolverWeights(
				raw=args.raw_weight,
				normalized=args.normalized_weight,
				total=args.total_weight,
			),
			max_distance=args.max_resolver_distance,
		)
		self.mutator = LocalInstructionMapMutator(
			self.config_generator,
			self.executor,
			mutation_budget=args.resolver_mutation_budget,
		)
		metadata = live_metadata(args, "live_two_step_tuning")
		metadata.update(
			{
				"instruction_space_source": "resolver_dataset_full",
				"shared_start_param_hash": hash_config(self.start_config),
				"resolver_mode": "database_only",
				"max_values_per_op": args.max_values_per_op,
				"opcodes": sorted(self.parameter_specs),
				"distance_metric": "euclidean_instruction_count_vector",
				"instruction_value_counts": {
					op: len(spec.values)
					for op, spec in sorted(self.parameter_specs.items())
				},
				"raw_weight": args.raw_weight,
				"normalized_weight": args.normalized_weight,
				"total_weight": args.total_weight,
				"max_resolver_distance": args.max_resolver_distance,
				"resolver_db_record_count": len(self.resolver_dataset.records),
				"resolver_mutation_budget": args.resolver_mutation_budget,
			}
		)
		self.recorder = ResultRecorder(args.output_dir, metadata)

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		seeds = [nearest_indices_from_counts(self.start_profile.raw_counts, self.parameter_specs)]
		return seeds

	def run(self, desired_result, input, limit):
		if self.recorder.iteration == 0:
			return self._run_start_config()
		requested_counts = dict(self.default_counts)
		requested_counts.update(
			counts_from_tuning_indices(desired_result.configuration.data, self.parameter_specs)
		)
		resolution = self.resolver.resolve(requested_counts)
		if not resolution.valid or resolution.profile is None:
			self.recorder.record(
				{
					"candidate_status": resolution.status,
					"runtime_ms": self.args.invalid_runtime_ms,
					"resolver_distance": resolution.distance,
					"raw_distance": resolution.raw_distance,
					"mix_distance": resolution.mix_distance,
					"total_distance": resolution.total_distance,
					"resolver_time_ms": resolution.resolver_time_ms,
					"duplicate_count": resolution.duplicate_count,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)

		refined = self.mutator.refine(requested_counts, resolution.profile)
		execution = self.executor.execute_config(refined.profile.config)
		if not execution.valid or execution.profile is None or execution.runtime_ms is None:
			self.recorder.record(
				{
					"candidate_status": "execution_failed",
					"runtime_ms": self.args.invalid_runtime_ms,
					"resolver_distance": refined.distance,
					"raw_distance": refined.distance,
					"mix_distance": resolution.mix_distance,
					"total_distance": resolution.total_distance,
					"resolver_time_ms": resolution.resolver_time_ms,
					"duplicate_count": resolution.duplicate_count,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)

		profile = execution.profile
		self.recorder.record(
			{
				"candidate_status": resolution.status if refined.source == "database" else "mutated_" + resolution.status,
				"runtime_ms": execution.runtime_ms,
				"exp_id": profile.exp_id,
				"param_hash": profile.param_hash,
				"resolver_distance": refined.distance,
				"raw_distance": refined.distance,
				"mix_distance": resolution.mix_distance,
				"total_distance": resolution.total_distance,
				"resolver_time_ms": resolution.resolver_time_ms,
				"duplicate_count": resolution.duplicate_count,
			}
		)
		return Result(time=execution.runtime_ms)

	def extra_convergence_criteria(self, result):
		limit = self.args.valid_config_limit
		return limit is not None and len(self.recorder.seen_exp_ids) >= limit

	def save_final_config(self, config):
		requested_counts = dict(self.default_counts)
		requested_counts.update(counts_from_tuning_indices(config.data, self.parameter_specs))
		resolution = self.resolver.resolve(requested_counts)
		refined = None
		if resolution.valid and resolution.profile is not None:
			refined = self.mutator.refine(requested_counts, resolution.profile)
		payload = {
			"encoding": "online_instruction_value_index",
			"index_config": dict(config.data),
			"requested_instruction_counts": requested_counts,
			"resolver_result": {
				"status": resolution.status,
				"distance": resolution.distance,
				"raw_distance": resolution.raw_distance,
				"mix_distance": resolution.mix_distance,
				"total_distance": resolution.total_distance,
				"duplicate_count": resolution.duplicate_count,
			},
			"resolved_config": None
			if refined is None
			else refined.profile.config,
			"db_match": None
			if resolution.profile is None
			else {
				"exp_id": resolution.profile.exp_id,
				"param_hash": resolution.profile.param_hash,
			},
			"mutation_result": None
			if refined is None
			else {
				"source": refined.source,
				"distance": refined.distance,
				"profile_count": len(refined.mutated_profiles),
				"chosen_param_hash": refined.profile.param_hash,
			},
		}
		(self.recorder.output_dir / "final_config.json").write_text(
			json.dumps(payload, indent=2, sort_keys=True)
		)
		self.recorder.write_summary()

	def _run_start_config(self):
		execution = self.executor.execute_config(self.start_config)
		if not execution.valid or execution.profile is None or execution.runtime_ms is None:
			self.recorder.record(
				{
					"candidate_status": "shared_start_failed",
					"runtime_ms": self.args.invalid_runtime_ms,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)
		profile = execution.profile
		self.recorder.record(
			{
				"candidate_status": "shared_start",
				"runtime_ms": execution.runtime_ms,
				"exp_id": profile.exp_id,
				"param_hash": profile.param_hash,
				"resolver_distance": 0.0,
				"raw_distance": 0.0,
				"mix_distance": 0.0,
				"total_distance": 0.0,
				"resolver_time_ms": 0.0,
				"duplicate_count": 1,
			}
		)
		return Result(time=execution.runtime_ms)

def build_argparser():
	parser = opentuner.default_argparser()
	add_live_arguments(parser)
	parser.add_argument("--max-values-per-op", type=int, default=12)
	parser.add_argument("--raw-weight", type=float, default=0.4)
	parser.add_argument("--normalized-weight", type=float, default=0.5)
	parser.add_argument("--total-weight", type=float, default=0.1)
	parser.add_argument("--max-resolver-distance", type=float, default=None)
	parser.add_argument("--resolver-mutation-budget", type=int, default=64)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	LiveTwoStepTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
