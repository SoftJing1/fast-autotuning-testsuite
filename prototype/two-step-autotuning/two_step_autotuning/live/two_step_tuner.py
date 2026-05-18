from __future__ import annotations

import json
from pathlib import Path
import time

import opentuner
from opentuner import MeasurementInterface, Result
from opentuner.tuningrunmain import TuningRunMain

from ..core.common_args import add_live_arguments, live_metadata
from ..core.instruction_space import (
	build_instruction_manipulator,
	build_instruction_parameter_specs_from_profiles,
	counts_from_tuning_indices,
	median_instruction_counts,
	nearest_indices_from_counts,
)
from ..core.resolver import LiveResolutionResult, OnlineInstructionMapResolver, ResolverWeights
from ..core.result_recorder import ResultRecorder
from ..core.instruction_map import canonical_config_key
from .config_space import LiveConfigGenerator
from .config_space import LiveConfigPool, LIVE_CONFIG_INDEX
from .executor import LiveKernelExecutor, check_instruction_counter_available


class _ResolverSearchInterface(MeasurementInterface):
	def __init__(
		self,
		args,
		pool: LiveConfigPool,
		executor: LiveKernelExecutor,
		scorer: OnlineInstructionMapResolver,
		requested_counts: dict[str, int],
		seed_values: list[dict[str, int]],
	):
		super().__init__(args)
		self.pool = pool
		self.executor = executor
		self.scorer = scorer
		self.requested_counts = requested_counts
		self.seed_values = seed_values
		self.best_resolution = None
		self._seen_keys: set[str] = set()
		self._collected_profiles = []

	def manipulator(self):
		return self.pool.manipulator()

	def seed_configurations(self):
		return self.seed_values

	def run(self, desired_result, input, limit):
		config = self.pool.config_from_values(desired_result.configuration.data)
		result = self.executor.profile_config(config)
		if not result.valid or result.profile is None:
			return Result(time=1.0e12)
		key = canonical_config_key(result.profile.config)
		if key not in self._seen_keys:
			self._seen_keys.add(key)
			self._collected_profiles.append(result.profile)
		resolution = self.scorer.score_profile(self.requested_counts, result.profile)
		if resolution.distance is not None and (
			self.best_resolution is None or resolution.distance < self.best_resolution.distance
		):
			self.best_resolution = resolution
		return Result(time=resolution.distance if resolution.distance is not None else 1.0e12)

	def save_final_config(self, config):
		return None


class _AutoTuningInstructionMapResolver:
	def __init__(
		self,
		output_dir: str | Path,
		config_generator: LiveConfigGenerator,
		executor: LiveKernelExecutor,
		weights: ResolverWeights,
		iteration_limit: int,
		max_distance: float | None = None,
		pool_size: int | None = None,
	):
		self.output_dir = Path(output_dir)
		self.config_generator = config_generator
		self.executor = executor
		self.iteration_limit = iteration_limit
		self.pool_size = pool_size if pool_size is not None else max(iteration_limit * 4, 128)
		self.selector = OnlineInstructionMapResolver(weights=weights, max_distance=max_distance)
		self._resolve_counter = 0

	def add_profiles(self, profiles):
		self.selector.add_profiles(profiles)

	def resolve(self, requested_counts: dict[str, int]):
		start = time.perf_counter()
		self._resolve_counter += 1
		pool = self._build_pool(requested_counts)
		seed_values = self._seed_values(requested_counts, pool)
		args = self._resolver_args(self._resolve_counter)
		interface = _ResolverSearchInterface(
			args=args,
			pool=pool,
			executor=self.executor,
			scorer=self.selector,
			requested_counts=requested_counts,
			seed_values=seed_values,
		)
		TuningRunMain(interface, args).main()
		if interface._collected_profiles:
			self.selector.add_profiles(interface._collected_profiles)
		best = interface.best_resolution
		total_ms = (time.perf_counter() - start) * 1000.0
		if best is None:
			return LiveResolutionResult("invalid", None, resolver_time_ms=total_ms)
		if best.distance is not None and self.selector.max_distance is not None and best.distance > self.selector.max_distance:
			return type(best)(
				status="invalid",
				profile=None,
				distance=best.distance,
				raw_distance=best.raw_distance,
				mix_distance=best.mix_distance,
				total_distance=best.total_distance,
				duplicate_count=best.duplicate_count,
				resolver_time_ms=total_ms,
			)
		return type(best)(
			status=best.status,
			profile=best.profile,
			distance=best.distance,
			raw_distance=best.raw_distance,
			mix_distance=best.mix_distance,
			total_distance=best.total_distance,
			duplicate_count=best.duplicate_count,
			resolver_time_ms=total_ms,
		)

	def _build_pool(self, requested_counts: dict[str, int]) -> LiveConfigPool:
		configs = []
		seen = set()
		seed_resolution = self.selector.resolve(requested_counts)
		if seed_resolution.valid and seed_resolution.profile is not None:
			key = canonical_config_key(seed_resolution.profile.config)
			seen.add(key)
			configs.append(seed_resolution.profile.config)
		for config in self.config_generator.generate(self.pool_size):
			key = canonical_config_key(config)
			if key in seen:
				continue
			seen.add(key)
			configs.append(config)
			if len(configs) >= self.pool_size:
				break
		return LiveConfigPool(
			kernel_type=self.config_generator.kernel_type,
			input_size=self.config_generator.input_size,
			configs=tuple(configs),
		)

	def _seed_values(self, requested_counts: dict[str, int], pool: LiveConfigPool) -> list[dict[str, int]]:
		seed_resolution = self.selector.resolve(requested_counts)
		if not seed_resolution.valid or seed_resolution.profile is None:
			return [{LIVE_CONFIG_INDEX: 0}] if pool.configs else []
		seed_key = canonical_config_key(seed_resolution.profile.config)
		for index, config in enumerate(pool.configs):
			if canonical_config_key(config) == seed_key:
				return [{LIVE_CONFIG_INDEX: index}]
		return [{LIVE_CONFIG_INDEX: 0}] if pool.configs else []

	def _resolver_args(self, resolve_index: int):
		resolver_run_dir = self.output_dir / "resolver_runs"
		resolver_run_dir.mkdir(parents=True, exist_ok=True)
		parser = opentuner.default_argparser()
		args = parser.parse_args(
			[
				"--test-limit",
				str(self.iteration_limit),
				"--database",
				f"sqlite:///{resolver_run_dir / f'resolve_{resolve_index:05d}.db'}",
				"--quiet",
				"--label",
				f"resolver_{resolve_index:05d}",
			]
		)
		return args


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
		self.executor = LiveKernelExecutor(
			args.output_dir,
			args.kernel,
			args.input_size,
			build_dir=args.build_dir,
			runs_per_config=args.runs_per_config,
		)
		self.bootstrap_profiles = self._profile_new_configs(args.bootstrap_profile_count)
		if not self.bootstrap_profiles:
			raise RuntimeError("bootstrap profiling produced no valid instruction maps")
		self.parameter_specs = build_instruction_parameter_specs_from_profiles(
			self.bootstrap_profiles,
			max_values_per_op=args.max_values_per_op,
		)
		if not self.parameter_specs:
			raise RuntimeError("bootstrap profiles produced no varying instruction-map parameters")
		self.default_counts = median_instruction_counts(self.bootstrap_profiles)
		self._manipulator = build_instruction_manipulator(self.parameter_specs)
		self.resolver = _AutoTuningInstructionMapResolver(
			output_dir=args.output_dir,
			config_generator=self.config_generator,
			executor=self.executor,
			weights=ResolverWeights(
				raw=args.raw_weight,
				normalized=args.normalized_weight,
				total=args.total_weight,
			),
			iteration_limit=args.resolver_candidate_limit,
			max_distance=args.max_resolver_distance,
		)
		self.resolver.add_profiles(self.bootstrap_profiles)
		metadata = live_metadata(args, "live_two_step_tuning")
		metadata.update(
			{
				"bootstrap_profile_count": len(self.bootstrap_profiles),
				"resolver_iteration_limit": args.resolver_candidate_limit,
				"max_values_per_op": args.max_values_per_op,
				"opcodes": sorted(self.parameter_specs),
				"instruction_value_counts": {
					op: len(spec.values)
					for op, spec in sorted(self.parameter_specs.items())
				},
				"raw_weight": args.raw_weight,
				"normalized_weight": args.normalized_weight,
				"total_weight": args.total_weight,
				"max_resolver_distance": args.max_resolver_distance,
				"resolver_bootstrap": "online_only",
			}
		)
		self.recorder = ResultRecorder(args.output_dir, metadata)

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		seeds = []
		for profile in self.bootstrap_profiles[: self.args.seed_record_count]:
			seeds.append(nearest_indices_from_counts(profile.raw_counts, self.parameter_specs))
		return seeds

	def run(self, desired_result, input, limit):
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

		execution = self.executor.execute_config(resolution.profile.config)
		if not execution.valid or execution.profile is None or execution.runtime_ms is None:
			self.recorder.record(
				{
					"candidate_status": "execution_failed",
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

		profile = execution.profile
		self.resolver.add_profiles([profile])
		self.recorder.record(
			{
				"candidate_status": resolution.status,
				"runtime_ms": execution.runtime_ms,
				"exp_id": profile.exp_id,
				"param_hash": profile.param_hash,
				"resolver_distance": resolution.distance,
				"raw_distance": resolution.raw_distance,
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
			"resolved_config": None if resolution.profile is None else resolution.profile.config,
			"db_match": None
			if resolution.profile is None
			else {
				"exp_id": resolution.profile.exp_id,
				"param_hash": resolution.profile.param_hash,
				"runtime_ms": resolution.profile.runtime_ms,
			},
		}
		(self.recorder.output_dir / "final_config.json").write_text(
			json.dumps(payload, indent=2, sort_keys=True)
		)
		self.recorder.write_summary()

	def _profile_new_configs(self, count: int):
		profiles = []
		if count <= 0:
			return profiles
		for config in self.config_generator.generate(count):
			result = self.executor.profile_config(config)
			if result.valid and result.profile is not None:
				profiles.append(result.profile)
		return profiles


def build_argparser():
	parser = opentuner.default_argparser()
	add_live_arguments(parser)
	parser.add_argument("--bootstrap-profile-count", type=int, default=32)
	parser.add_argument(
		"--resolver-candidate-limit",
		type=int,
		default=64,
		help="Inner resolver autotuning iteration limit per requested instruction map.",
	)
	parser.add_argument("--max-values-per-op", type=int, default=12)
	parser.add_argument("--raw-weight", type=float, default=0.4)
	parser.add_argument("--normalized-weight", type=float, default=0.5)
	parser.add_argument("--total-weight", type=float, default=0.1)
	parser.add_argument("--max-resolver-distance", type=float, default=None)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	LiveTwoStepTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
