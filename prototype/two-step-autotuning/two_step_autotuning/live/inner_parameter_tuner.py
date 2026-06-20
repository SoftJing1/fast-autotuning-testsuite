from __future__ import annotations

from argparse import Namespace
import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
import time

from opentuner import MeasurementInterface, Result
from opentuner.tuningrunmain import TuningRunMain

from ..core.instruction_space import filter_instruction_opcodes
from ..core.parameter_space import (
	build_generator_parameter_index_specs,
	build_parameter_manipulator,
	config_from_parameter_indices,
	indices_from_config,
)
from ..core.types import LiveProfile
from .config_space import LiveConfigGenerator
from .executor import LiveKernelExecutor


def instruction_map_distance(
	requested_counts: dict[str, Any],
	candidate_counts: dict[str, Any],
	normalization_scales: dict[str, float] | None = None,
) -> float:
	opcodes = (
		filter_instruction_opcodes(normalization_scales)
		if normalization_scales is not None
		else filter_instruction_opcodes(set(requested_counts) | set(candidate_counts))
	)
	return sum(
		(
			(
				float(requested_counts.get(op, 0)) - float(candidate_counts.get(op, 0))
			)
			/ max(float((normalization_scales or {}).get(op, 1.0)), 1.0)
		)
		** 2
		for op in opcodes
	) ** 0.5


def target_distance_for_ratio(db_distance: float | None, target_ratio: float) -> float | None:
	if db_distance is None:
		return None
	return max(float(db_distance), 0.0) * max(float(target_ratio), 0.0)


def target_hit(distance: float | None, target_distance: float | None) -> bool:
	if distance is None or target_distance is None:
		return False
	return float(distance) <= float(target_distance)


@dataclass(frozen=True)
class InnerParameterTunerResult:
	profile: LiveProfile
	distance: float
	source: str
	db_distance: float | None
	distance_ratio: float | None
	target_distance: float | None
	target_hit: bool
	valid_profile_count: int
	total_tests: int
	invalid_count: int
	wall_time_ms: float


class InnerParameterTuningInterface(MeasurementInterface):
	def __init__(
		self,
		args,
		config_generator: LiveConfigGenerator,
		executor: LiveKernelExecutor,
		requested_counts: dict[str, int],
		db_profile: LiveProfile,
		db_distance: float | None,
		normalization_scales: dict[str, float] | None = None,
	):
		ensure_inner_opentuner_args(args)
		super().__init__(args)
		self.config_generator = config_generator
		self.executor = executor
		self.requested_counts = dict(requested_counts)
		self.db_profile = db_profile
		self.db_distance = db_distance
		self.normalization_scales = normalization_scales
		self.parameter_specs = build_generator_parameter_index_specs(
			config_generator.kernel_type,
			config_generator.input_size,
			device_type=config_generator.device_type,
		)
		self.fixed_config = config_generator.fixed_input_config()
		self._manipulator = build_parameter_manipulator(self.parameter_specs)
		self.invalid_distance_penalty = float(args.resolver_inner_invalid_distance_penalty)
		self.target_distance = target_distance_for_ratio(
			db_distance,
			float(args.resolver_inner_target_ratio),
		)
		self.valid_profile_count = 0
		self.total_tests = 0
		self.invalid_count = 0
		self.best_profile: LiveProfile | None = None
		self.best_distance: float | None = None

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		return [indices_from_config(self.db_profile.config, self.parameter_specs)]

	def run(self, desired_result, input=None, limit=None):
		self.total_tests += 1
		index_config = dict(desired_result.configuration.data)
		config = config_from_parameter_indices(
			index_config,
			self.parameter_specs,
			fixed_config=self.fixed_config,
		)
		if not self.config_generator.validate_config(config):
			self.invalid_count += 1
			return Result(time=self.invalid_distance_penalty)

		result = self.executor.profile_config(config)
		if not result.valid or result.profile is None:
			self.invalid_count += 1
			return Result(time=self.invalid_distance_penalty)

		self.valid_profile_count += 1
		distance = instruction_map_distance(
			self.requested_counts,
			result.profile.raw_counts,
			self.normalization_scales,
		)
		if self.best_distance is None or distance < self.best_distance:
			self.best_profile = result.profile
			self.best_distance = distance
		return Result(time=distance)

	def extra_convergence_criteria(self, result):
		if self.valid_profile_count >= int(self.args.resolver_inner_valid_profile_limit):
			return True
		if self.total_tests >= int(self.args.resolver_inner_test_limit):
			return True
		return target_hit(self.best_distance, self.target_distance)

	def tune(self) -> InnerParameterTunerResult:
		start = time.perf_counter()
		TuningRunMain(self, self.args).main()
		return self._result(start)

	def _result(self, start: float) -> InnerParameterTunerResult:
		if self.best_profile is None or self.best_distance is None:
			self.best_profile = self.db_profile
			self.best_distance = (
				float(self.db_distance)
				if self.db_distance is not None
				else instruction_map_distance(
					self.requested_counts,
					self.db_profile.raw_counts,
					self.normalization_scales,
				)
			)
		distance_ratio = None
		if self.db_distance is not None:
			if self.db_distance == 0.0:
				distance_ratio = 1.0 if self.best_distance == 0.0 else None
			else:
				distance_ratio = self.best_distance / self.db_distance
		source = "inner_tuner" if self.best_profile.param_hash != self.db_profile.param_hash else "database"
		return InnerParameterTunerResult(
			profile=self.best_profile,
			distance=self.best_distance,
			source=source,
			db_distance=self.db_distance,
			distance_ratio=distance_ratio,
			target_distance=self.target_distance,
			target_hit=target_hit(self.best_distance, self.target_distance),
			valid_profile_count=self.valid_profile_count,
			total_tests=self.total_tests,
			invalid_count=self.invalid_count,
			wall_time_ms=(time.perf_counter() - start) * 1000.0,
		)


def ensure_inner_opentuner_args(args) -> None:
	defaults = {
		"bail_threshold": 500,
		"database": None,
		"display_frequency": 10,
		"generate_bandit_technique": False,
		"label": None,
		"list_techniques": False,
		"machine_class": None,
		"no_dups": False,
		"parallel_compile": False,
		"parallelism": 1,
		"pipelining": 0,
		"print_params": False,
		"print_search_space_size": False,
		"quiet": True,
		"results_log": None,
		"results_log_details": None,
		"seed_configuration": [],
		"stop_after": None,
		"technique": None,
		"test_limit": None,
	}
	for name, value in defaults.items():
		if not hasattr(args, name):
			setattr(args, name, copy.deepcopy(value))
	if args.database is None:
		output_dir = Path(tempfile.mkdtemp(prefix="two_step_inner_opentuner_"))
		args.database = f"sqlite:///{output_dir / 'opentuner.db'}"


def build_inner_opentuner_args(args, db_profile: LiveProfile) -> Namespace:
	values = dict(vars(args))
	inner_args = Namespace(**values)
	ensure_inner_opentuner_args(inner_args)
	inner_args.parallelism = 1
	inner_args.parallel_compile = False
	inner_args.test_limit = int(args.resolver_inner_test_limit)
	inner_args.no_dups = True
	inner_args.seed_configuration = []
	inner_args.label = f"inner-{db_profile.param_hash}"
	output_dir = Path(getattr(args, "output_dir", "/tmp")) / "_inner_parameter_tuner" / db_profile.param_hash
	output_dir.mkdir(parents=True, exist_ok=True)
	inner_args.database = f"sqlite:///{output_dir / 'opentuner.db'}"
	return inner_args


def tune_inner_parameter_config(
	args,
	config_generator: LiveConfigGenerator,
	executor: LiveKernelExecutor,
	requested_counts: dict[str, int],
	db_profile: LiveProfile,
	db_distance: float | None,
	normalization_scales: dict[str, float] | None = None,
) -> InnerParameterTunerResult:
	inner_args = build_inner_opentuner_args(args, db_profile)
	return InnerParameterTuningInterface(
		inner_args,
		config_generator,
		executor,
		requested_counts,
		db_profile,
		db_distance,
		normalization_scales=normalization_scales,
	).tune()
