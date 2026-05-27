from __future__ import annotations

import argparse
import json

import opentuner
from opentuner import MeasurementInterface, Result

from ..core.common_args import add_common_arguments, warn_if_seed_budget_exhausts_search
from ..core.dataset import TuningDataset
from ..core.parameter_distance import PARAMETER_DISTANCE_METRICS
from ..core.parameter_resolver import DatabaseApproxParameterResolver
from ..core.parameter_space import (
	build_parameter_index_specs,
	build_parameter_manipulator,
	config_from_tuning_indices,
	indices_from_config,
)
from ..core.resolver import ExactConfigLookup
from ..core.result_recorder import ResultRecorder
from ..core.types import ResolutionResult


class ParameterTuningInterface(MeasurementInterface):
	def __init__(self, args):
		super().__init__(args)
		warn_if_seed_budget_exhausts_search(args, "parameter_tuning")
		self.dataset = TuningDataset(args.db, args.kernel, args.input_size)
		self.parameter_specs = build_parameter_index_specs(self.dataset)
		self.lookup = ExactConfigLookup(self.dataset)
		self.approx_resolver = (
			None
			if args.parameter_resolver == "exact"
			else DatabaseApproxParameterResolver(
				self.dataset,
				self.parameter_specs,
				metric=args.parameter_distance_metric,
				max_distance=args.parameter_max_distance,
			)
		)
		self._manipulator = build_parameter_manipulator(self.parameter_specs)
		self.recorder = ResultRecorder(
			args.output_dir,
			{
				"method": "parameter_tuning",
				"db": args.db,
				"kernel": args.kernel,
				"input_size": args.input_size,
				"invalid_runtime_ms": args.invalid_runtime_ms,
				"seed_record_count": args.seed_record_count,
				"random_seed": args.random_seed,
				"valid_config_limit": args.valid_config_limit,
				"record_count": len(self.dataset.records),
				"encoding": "observed_value_index",
				"resolver": args.parameter_resolver,
				"parameter_distance_metric": args.parameter_distance_metric
				if args.parameter_resolver == "approx"
				else None,
				"parameter_max_distance": args.parameter_max_distance
				if args.parameter_resolver == "approx"
				else None,
				"parameter_value_counts": {
					name: len(spec.values)
					for name, spec in sorted(self.parameter_specs.items())
				},
			},
		)

	def manipulator(self):
		return self._manipulator

	def seed_configurations(self):
		return [
			indices_from_config(record.config, self.parameter_specs)
			for record in self.dataset.sampled_records(self.args.seed_record_count, self.args.random_seed)
		]

	def _resolve_config(self, config: dict) -> ResolutionResult:
		if self.args.parameter_resolver == "exact":
			record = self.lookup.resolve(config)
			return ResolutionResult(status="valid" if record is not None else "invalid", record=record)
		if self.approx_resolver is None:
			raise RuntimeError("approximate parameter resolver was not initialized")
		return self.approx_resolver.resolve(config)

	def run(self, desired_result, input, limit):
		values = desired_result.configuration.data
		config = config_from_tuning_indices(self.dataset, values, self.parameter_specs)
		result = self._resolve_config(config)
		if not result.valid:
			self.recorder.record(
				{
					"candidate_status": "invalid",
					"runtime_ms": self.args.invalid_runtime_ms,
					"resolver_distance": result.distance,
					"resolver_time_ms": result.resolver_time_ms,
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
		decoded_config = config_from_tuning_indices(self.dataset, index_config, self.parameter_specs)
		result = self._resolve_config(decoded_config)
		record = result.record
		payload = {
			"encoding": "observed_value_index",
			"index_config": index_config,
			"decoded_config": decoded_config,
			"resolver": self.args.parameter_resolver,
			"resolver_status": result.status,
			"resolver_distance": result.distance,
			"db_match": None
			if record is None
			else {
				"exp_id": record.exp_id,
				"param_hash": record.param_hash,
				"runtime_ms": record.runtime_ms,
			},
		}
		(self.recorder.output_dir / "final_config.json").write_text(
			json.dumps(payload, indent=2, sort_keys=True)
		)
		self.recorder.write_summary()


def build_argparser():
	parser = opentuner.default_argparser()
	add_common_arguments(parser)
	parser.add_argument(
		"--parameter-resolver",
		choices=("approx", "exact"),
		default="approx",
		help="Use exact DB lookup or resolve proposed configs to the nearest DB row.",
	)
	parser.add_argument(
		"--parameter-distance-metric",
		choices=PARAMETER_DISTANCE_METRICS,
		default="euclidean",
		help="Distance metric used by --parameter-resolver approx.",
	)
	parser.add_argument(
		"--parameter-max-distance",
		type=float,
		default=None,
		help="Optional invalid threshold for approximate parameter resolution.",
	)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	ParameterTuningInterface.main(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
