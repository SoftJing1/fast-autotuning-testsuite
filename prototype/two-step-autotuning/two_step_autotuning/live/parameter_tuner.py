from __future__ import annotations

import json

import opentuner
from opentuner import MeasurementInterface, Result

from ..core.common_args import add_live_arguments, live_metadata
from ..core.result_recorder import ResultRecorder
from .config_space import LiveConfigGenerator
from .executor import LiveKernelExecutor


class LiveParameterTuningInterface(MeasurementInterface):
	def __init__(self, args):
		super().__init__(args)
		if args.valid_config_limit is not None and args.candidate_pool_size < args.valid_config_limit:
			raise ValueError("--candidate-pool-size must be >= --valid-config-limit")
		self.config_generator = LiveConfigGenerator(
			args.kernel,
			args.input_size,
			device_type=args.device_type,
			random_seed=args.random_seed,
		)
		self.pool = self.config_generator.build_pool(args.candidate_pool_size)
		self.executor = LiveKernelExecutor(
			args.output_dir,
			args.kernel,
			args.input_size,
			build_dir=args.build_dir,
			runs_per_config=args.runs_per_config,
		)
		metadata = live_metadata(args, "live_parameter_tuning")
		metadata.update(
			{
				"candidate_pool_size": len(self.pool.configs),
				"parameter_encoding": "online_valid_config_index",
			}
		)
		self.recorder = ResultRecorder(args.output_dir, metadata)
		self._manipulator = self.pool.manipulator()

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		return self.pool.seed_configurations(self.args.seed_record_count)

	def run(self, desired_result, input, limit):
		config = self.pool.config_from_values(desired_result.configuration.data)
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
		limit = self.args.valid_config_limit
		return limit is not None and len(self.recorder.seen_exp_ids) >= limit

	def save_final_config(self, config):
		decoded_config = self.pool.config_from_values(config.data)
		result = self.executor.execute_config(decoded_config)
		payload = {
			"encoding": "online_valid_config_index",
			"index_config": dict(config.data),
			"decoded_config": decoded_config,
			"result": None
			if result.profile is None
			else {
				"exp_id": result.profile.exp_id,
				"param_hash": result.profile.param_hash,
				"runtime_ms": result.runtime_ms,
			},
		}
		(self.recorder.output_dir / "final_config.json").write_text(
			json.dumps(payload, indent=2, sort_keys=True)
		)
		self.recorder.write_summary()


def build_argparser():
	parser = opentuner.default_argparser()
	add_live_arguments(parser)
	parser.add_argument(
		"--candidate-pool-size",
		type=int,
		default=1024,
		help="Number of online-generated valid configs available to the live parameter tuner.",
	)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	LiveParameterTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
