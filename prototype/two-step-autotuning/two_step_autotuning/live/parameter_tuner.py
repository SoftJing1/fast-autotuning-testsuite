from __future__ import annotations

import json

import opentuner
from opentuner import MeasurementInterface, Result

from ..core.common_args import add_live_arguments, live_metadata
from ..core.parameter_space import (
	build_generator_parameter_index_specs,
	build_parameter_manipulator,
	config_from_parameter_indices,
	indices_from_config,
)
from ..core.result_recorder import ResultRecorder
from .config_space import LiveConfigGenerator, hash_config
from .executor import LiveKernelExecutor


class LiveParameterTuningInterface(MeasurementInterface):
	def __init__(self, args):
		super().__init__(args)
		self.config_generator = LiveConfigGenerator(
			args.kernel,
			args.input_size,
			device_type=args.device_type,
			random_seed=args.random_seed,
		)
		self.start_config = self.config_generator.generate(1)[0]
		self.parameter_specs = build_generator_parameter_index_specs(
			args.kernel,
			args.input_size,
			device_type=args.device_type,
		)
		self.executor = LiveKernelExecutor(
			args.output_dir,
			args.kernel,
			args.input_size,
			build_dir=args.build_dir,
			warmup_runs=args.warmup_runs,
			runs_per_config=args.runs_per_config,
		)
		metadata = live_metadata(args, "live_parameter_tuning")
		metadata.update(
			{
				"parameter_encoding": "generator_value_index",
				"parameter_space_source": "generator_definition",
				"shared_start_param_hash": hash_config(self.start_config),
				"parameter_value_counts": {
					name: len(spec.values)
					for name, spec in sorted(self.parameter_specs.items())
				},
			}
		)
		self.recorder = ResultRecorder(args.output_dir, metadata)
		self._manipulator = build_parameter_manipulator(self.parameter_specs)

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		seeds = [indices_from_config(self.start_config, self.parameter_specs)]
		for config in self.config_generator.generate(max(0, self.args.seed_record_count - 1)):
			seeds.append(indices_from_config(config, self.parameter_specs))
		return seeds[: self.args.seed_record_count]

	def run(self, desired_result, input, limit):
		config = config_from_parameter_indices(
			desired_result.configuration.data,
			self.parameter_specs,
			fixed_config=self.config_generator.fixed_input_config(),
		)
		if not self.config_generator.validate_config(config):
			self.recorder.record(
				{
					"candidate_status": "invalid_precheck",
					"runtime_ms": self.args.invalid_runtime_ms,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)
		result = self.executor.execute_config(config)
		if not result.valid or result.profile is None or result.runtime_ms is None:
			self.recorder.record(
				{
					"candidate_status": "invalid",
					"runtime_ms": self.args.invalid_runtime_ms,
				}
			)
			return Result(time=self.args.invalid_runtime_ms)
		profile = result.profile
		self.recorder.record(
			{
				"candidate_status": "valid",
				"runtime_ms": result.runtime_ms,
				"exp_id": profile.exp_id,
				"param_hash": profile.param_hash,
			}
		)
		return Result(time=result.runtime_ms)

	def extra_convergence_criteria(self, result):
		limit = self.args.valid_evaluation_limit
		return limit is not None and self.recorder.valid_evaluation_count >= limit

	def save_final_config(self, config):
		index_config = dict(config.data)
		decoded_config = config_from_parameter_indices(
			index_config,
			self.parameter_specs,
			fixed_config=self.config_generator.fixed_input_config(),
		)
		if not self.config_generator.validate_config(decoded_config):
			payload = {
				"encoding": "generator_value_index",
				"index_config": index_config,
				"decoded_config": decoded_config,
				"result": None,
				"validation": "invalid_precheck",
			}
			(self.recorder.output_dir / "final_config.json").write_text(
				json.dumps(payload, indent=2, sort_keys=True)
			)
			self.recorder.write_summary()
			return
		result = self.executor.execute_config(decoded_config)
		payload = {
			"encoding": "generator_value_index",
			"index_config": index_config,
			"decoded_config": decoded_config,
			"result": None
			if result.profile is None
			else {
				"exp_id": result.profile.exp_id,
				"param_hash": result.profile.param_hash,
				"runtime_ms": result.runtime_ms,
				"warmup_runtimes_ms": list(result.warmup_runtimes_ms),
				"measured_runtimes_ms": list(result.measured_runtimes_ms),
			},
		}
		(self.recorder.output_dir / "final_config.json").write_text(
			json.dumps(payload, indent=2, sort_keys=True)
		)
		self.recorder.write_summary()


def build_argparser():
	parser = opentuner.default_argparser()
	add_live_arguments(parser)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	LiveParameterTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
