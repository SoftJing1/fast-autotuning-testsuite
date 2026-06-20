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
from ..core.resolver import (
	DatabaseApproxInstructionMapResolver,
	INSTRUCTION_DISTANCE_METRICS,
	LiveResolutionResult,
)
from ..core.result_recorder import ResultRecorder
from ..core.types import LiveProfile
from .config_space import LiveConfigGenerator, hash_config
from .executor import LiveKernelExecutor, check_instruction_counter_available
from .inner_parameter_tuner import (
	InnerParameterTunerResult,
	instruction_map_distance,
	target_distance_for_ratio,
	target_hit,
	tune_inner_parameter_config,
)


def resolver_mode_for_refinement_mode(refinement_mode: str) -> str:
	return "database_plus_inner_tuner" if refinement_mode == "inner-tuner" else "database_only"


def resolver_distance_ratio(
	db_distance: float | None,
	refined_distance: float | None,
) -> float | None:
	if db_distance is None or refined_distance is None:
		return None
	if db_distance == 0.0:
		return 1.0 if refined_distance == 0.0 else None
	return refined_distance / db_distance


class _DatabaseBackedLiveResolver:
	def __init__(
		self,
		dataset: TuningDataset,
		max_distance: float | None = None,
		distance_metric: str = "euclidean",
	):
		self.dataset = dataset
		self.resolver = DatabaseApproxInstructionMapResolver(
			dataset,
			max_distance=max_distance,
			metric=distance_metric,
		)

	def resolve(self, requested_counts: dict[str, int]):
		result = self.resolver.resolve(requested_counts)
		if not result.valid or result.record is None:
			return LiveResolutionResult(
				status=result.status,
				profile=None,
				distance=result.distance,
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
			require_runtime=False,
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
			max_distance=args.max_resolver_distance,
			distance_metric=args.resolver_distance_metric,
		)
		metadata = live_metadata(args, "live_two_step_tuning")
		metadata.update(
			{
				"instruction_space_source": "resolver_dataset_full",
				"shared_start_param_hash": hash_config(self.start_config),
				"resolver_mode": resolver_mode_for_refinement_mode(args.resolver_refinement_mode),
				"max_values_per_op": args.max_values_per_op,
				"opcodes": sorted(self.parameter_specs),
				"distance_metric": args.resolver_distance_metric,
				"instruction_value_counts": {
					op: len(spec.values)
					for op, spec in sorted(self.parameter_specs.items())
				},
				"max_resolver_distance": args.max_resolver_distance,
				"resolver_db_record_count": len(self.resolver_dataset.records),
				"resolver_refinement_mode": args.resolver_refinement_mode,
				"resolver_inner_valid_profile_limit": args.resolver_inner_valid_profile_limit,
				"resolver_inner_test_limit": args.resolver_inner_test_limit,
				"resolver_inner_target_ratio": args.resolver_inner_target_ratio,
				"resolver_inner_invalid_distance_penalty": args.resolver_inner_invalid_distance_penalty,
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
					"db_resolver_distance": resolution.distance,
					"refined_resolver_distance": "",
					"distance_ratio": "",
					"resolver_source": "database",
					"db_param_hash": "",
					"resolved_param_hash": "",
					"resolver_profile_count": 0,
					"resolver_time_ms": resolution.resolver_time_ms,
					"resolver_refinement_time_ms": 0.0,
					"duplicate_count": resolution.duplicate_count,
					"inner_valid_profile_count": 0,
					"inner_total_tests": 0,
					"inner_invalid_count": 0,
					"inner_target_distance": "",
					"inner_target_hit": "",
					"inner_wall_time_ms": 0.0,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)

		refined = self._refine_resolution(requested_counts, resolution.profile, resolution.distance)
		execution = self.executor.execute_config(refined.profile.config)
		if not execution.valid or execution.profile is None or execution.runtime_ms is None:
			self.recorder.record(
				{
					"candidate_status": "execution_failed",
					"runtime_ms": self.args.invalid_runtime_ms,
					"resolver_distance": refined.distance,
					"db_resolver_distance": resolution.distance,
					"refined_resolver_distance": refined.distance,
					"distance_ratio": refined.distance_ratio,
					"resolver_source": refined.source,
					"db_param_hash": resolution.profile.param_hash,
					"resolved_param_hash": refined.profile.param_hash,
					"resolver_profile_count": refined.valid_profile_count,
					"resolver_time_ms": resolution.resolver_time_ms,
					"resolver_refinement_time_ms": refined.wall_time_ms,
					"duplicate_count": resolution.duplicate_count,
					"inner_valid_profile_count": refined.valid_profile_count,
					"inner_total_tests": refined.total_tests,
					"inner_invalid_count": refined.invalid_count,
					"inner_target_distance": "" if refined.target_distance is None else refined.target_distance,
					"inner_target_hit": refined.target_hit,
					"inner_wall_time_ms": refined.wall_time_ms,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)

		profile = execution.profile
		self.recorder.record(
			{
				"candidate_status": resolution.status if refined.source == "database" else "inner_tuned_" + resolution.status,
				"runtime_ms": execution.runtime_ms,
				"exp_id": profile.exp_id,
				"param_hash": profile.param_hash,
				"resolver_distance": refined.distance,
				"db_resolver_distance": resolution.distance,
				"refined_resolver_distance": refined.distance,
				"distance_ratio": refined.distance_ratio,
				"resolver_source": refined.source,
				"db_param_hash": resolution.profile.param_hash,
				"resolved_param_hash": refined.profile.param_hash,
				"resolver_profile_count": refined.valid_profile_count,
				"resolver_time_ms": resolution.resolver_time_ms,
				"resolver_refinement_time_ms": refined.wall_time_ms,
				"duplicate_count": resolution.duplicate_count,
				"inner_valid_profile_count": refined.valid_profile_count,
				"inner_total_tests": refined.total_tests,
				"inner_invalid_count": refined.invalid_count,
				"inner_target_distance": "" if refined.target_distance is None else refined.target_distance,
				"inner_target_hit": refined.target_hit,
				"inner_wall_time_ms": refined.wall_time_ms,
			}
		)
		return Result(time=execution.runtime_ms)

	def extra_convergence_criteria(self, result):
		limit = self.args.valid_evaluation_limit
		return limit is not None and self.recorder.valid_evaluation_count >= limit

	def save_final_config(self, config):
		requested_counts = dict(self.default_counts)
		requested_counts.update(counts_from_tuning_indices(config.data, self.parameter_specs))
		resolution = self.resolver.resolve(requested_counts)
		refined = None
		if resolution.valid and resolution.profile is not None:
			refined = self._refine_resolution(requested_counts, resolution.profile, resolution.distance)
		payload = {
			"encoding": "online_instruction_value_index",
			"index_config": dict(config.data),
			"requested_instruction_counts": requested_counts,
			"resolver_result": {
				"status": resolution.status,
				"distance": resolution.distance,
				"duplicate_count": resolution.duplicate_count,
			},
			"refined_resolver_result": None
			if refined is None
			else {
				"source": refined.source,
				"distance": refined.distance,
				"distance_ratio": refined.distance_ratio,
				"profile_count": refined.valid_profile_count,
				"total_tests": refined.total_tests,
				"invalid_count": refined.invalid_count,
				"target_distance": refined.target_distance,
				"target_hit": refined.target_hit,
				"wall_time_ms": refined.wall_time_ms,
				"chosen_param_hash": refined.profile.param_hash,
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
			"inner_tuner_result": None
			if refined is None
			else {
				"source": refined.source,
				"distance": refined.distance,
				"distance_ratio": refined.distance_ratio,
				"valid_profile_count": refined.valid_profile_count,
				"total_tests": refined.total_tests,
				"invalid_count": refined.invalid_count,
				"target_distance": refined.target_distance,
				"target_hit": refined.target_hit,
				"wall_time_ms": refined.wall_time_ms,
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
				"db_resolver_distance": 0.0,
				"refined_resolver_distance": 0.0,
				"distance_ratio": 1.0,
				"resolver_source": "shared_start",
				"db_param_hash": profile.param_hash,
				"resolved_param_hash": profile.param_hash,
				"resolver_profile_count": 0,
				"resolver_time_ms": 0.0,
				"resolver_refinement_time_ms": 0.0,
				"duplicate_count": 1,
				"inner_valid_profile_count": 0,
				"inner_total_tests": 0,
				"inner_invalid_count": 0,
				"inner_target_distance": "",
				"inner_target_hit": "",
				"inner_wall_time_ms": 0.0,
			}
		)
		return Result(time=execution.runtime_ms)

	def _refine_resolution(self, requested_counts, db_profile, db_distance):
		normalization_scales = (
			self.resolver.resolver.normalization_scales
			if self.args.resolver_distance_metric == "normalized-euclidean"
			else None
		)
		if self.args.resolver_refinement_mode == "database":
			distance = (
				float(db_distance)
				if db_distance is not None
				else instruction_map_distance(requested_counts, db_profile.raw_counts, normalization_scales)
			)
			target_distance = target_distance_for_ratio(db_distance, self.args.resolver_inner_target_ratio)
			return InnerParameterTunerResult(
				profile=db_profile,
				distance=distance,
				source="database",
				db_distance=db_distance,
				distance_ratio=resolver_distance_ratio(db_distance, distance),
				target_distance=target_distance,
				target_hit=target_hit(distance, target_distance),
				valid_profile_count=0,
				total_tests=0,
				invalid_count=0,
				wall_time_ms=0.0,
			)
		return tune_inner_parameter_config(
			self.args,
			self.config_generator,
			self.executor,
			requested_counts,
			db_profile,
			db_distance,
			normalization_scales=normalization_scales,
		)

def build_argparser():
	parser = opentuner.default_argparser()
	add_live_arguments(parser)
	parser.add_argument("--max-values-per-op", type=int, default=12)
	parser.add_argument("--max-resolver-distance", type=float, default=None)
	parser.add_argument(
		"--resolver-distance-metric",
		choices=INSTRUCTION_DISTANCE_METRICS,
		default="euclidean",
		help="Distance metric used by the instruction-map resolver.",
	)
	parser.add_argument(
		"--resolver-refinement-mode",
		choices=("database", "inner-tuner"),
		default="inner-tuner",
	)
	parser.add_argument("--resolver-inner-valid-profile-limit", type=int, default=32)
	parser.add_argument("--resolver-inner-test-limit", type=int, default=512)
	parser.add_argument("--resolver-inner-target-ratio", type=float, default=0.1)
	parser.add_argument("--resolver-inner-invalid-distance-penalty", type=float, default=1.0e12)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	LiveTwoStepTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
