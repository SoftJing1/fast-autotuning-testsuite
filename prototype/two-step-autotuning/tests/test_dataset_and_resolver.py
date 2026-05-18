from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "prototype" / "two-step-autotuning"
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))
if str(PACKAGE_ROOT) not in sys.path:
	sys.path.insert(0, str(PACKAGE_ROOT))

from two_step_autotuning.core.dataset import TuningDataset
from two_step_autotuning.core.instruction_distance import InstructionDistanceModel
from two_step_autotuning.core.instruction_map import canonical_config_key
from two_step_autotuning.core.instruction_space import (
	build_instruction_parameter_specs_from_profiles,
	counts_from_tuning_indices,
	median_instruction_counts,
	nearest_indices_from_counts,
)
from two_step_autotuning.core.parameter_space import (
	build_parameter_index_specs,
	build_parameter_manipulator,
	config_from_tuning_indices,
	indices_from_config,
)
from two_step_autotuning.core.parameter_resolver import DatabaseApproxParameterResolver
from two_step_autotuning.core.resolver import DatabaseApproxInstructionMapResolver
from two_step_autotuning.core.types import LiveProfile
from scripts.internal.db_manager import ExperimentDB


DB_PATH = ROOT / "experiments" / "exp_20260415_large_scale_5000cfg" / "experiments.db"


def test_dataset_loads_completed_gaussian_records():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")

	assert len(dataset.records) == 5000
	assert dataset.opcodes
	assert dataset.best_record.runtime_ms > 0


def test_exact_config_lookup_uses_full_observed_config():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	record = dataset.records[0]
	specs = build_parameter_index_specs(dataset)
	values = indices_from_config(record.config, specs)

	reconstructed = config_from_tuning_indices(dataset, values, specs)

	assert canonical_config_key(reconstructed) == canonical_config_key(record.config)
	assert dataset.lookup_config(reconstructed).exp_id == record.exp_id


def test_parameter_index_manipulator_decodes_to_host_config_format():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	specs = build_parameter_index_specs(dataset)
	manipulator = build_parameter_manipulator(specs)

	for _ in range(20):
		index_config = manipulator.random()
		decoded = config_from_tuning_indices(dataset, index_config, specs)

		assert all(not key.startswith("idx.") for key in decoded)
		assert {"input_size_h", "input_size_w"}.issubset(decoded)
		for name, spec in specs.items():
			assert decoded[name] in spec.values


def test_instruction_resolver_returns_existing_record_for_exact_map():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	record = dataset.records[10]
	resolver = DatabaseApproxInstructionMapResolver(dataset)

	result = resolver.resolve(record.raw_counts)

	assert result.valid
	assert result.record is not None
	assert result.record.exp_id == record.exp_id
	assert result.distance == 0.0


def test_instruction_distance_model_finds_zero_distance_for_same_record():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	model = InstructionDistanceModel(dataset)

	distances, raw_distances, mix_distances, total_distances = model.component_distances_to_counts(
		dataset.records[10].raw_counts
	)

	assert distances[10] == 0.0
	assert raw_distances[10] == 0.0
	assert mix_distances[10] == 0.0
	assert total_distances[10] == 0.0


def test_parameter_resolver_returns_existing_record_for_exact_config():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	specs = build_parameter_index_specs(dataset)
	record = dataset.records[10]
	resolver = DatabaseApproxParameterResolver(dataset, specs, metric="ordinal")

	result = resolver.resolve(record.config)

	assert result.valid
	assert result.record is not None
	assert result.record.exp_id == record.exp_id
	assert result.distance == 0.0


def test_parameter_resolver_maps_random_index_config_to_dataset_record():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	specs = build_parameter_index_specs(dataset)
	manipulator = build_parameter_manipulator(specs)
	resolver = DatabaseApproxParameterResolver(dataset, specs, metric="ordinal")
	decoded = config_from_tuning_indices(dataset, manipulator.random(), specs)

	result = resolver.resolve(decoded)

	assert result.valid
	assert result.record is not None
	assert dataset.lookup_config(result.record.config).exp_id == result.record.exp_id


def test_instruction_space_encodes_live_representative_values():
	profiles = [
		LiveProfile(1, "gaussian", "512x512", "a", {"x": 1}, {"add": 10, "mul": 2}),
		LiveProfile(2, "gaussian", "512x512", "b", {"x": 2}, {"add": 20, "mul": 2}),
		LiveProfile(3, "gaussian", "512x512", "c", {"x": 3}, {"add": 30, "mul": 4}),
	]

	specs = build_instruction_parameter_specs_from_profiles(profiles, max_values_per_op=3)
	assert specs["add"].values == (10, 20, 30)
	assert specs["mul"].values == (2, 4)

	indices = nearest_indices_from_counts({"add": 21, "mul": 4}, specs)
	assert counts_from_tuning_indices(indices, specs) == {"add": 20, "mul": 4}
	assert median_instruction_counts(profiles) == {"add": 20, "mul": 2}


def test_experiment_db_can_store_profiled_rows_by_key(tmp_path):
	db_path = tmp_path / "live.db"
	with ExperimentDB(str(db_path)) as db:
		exp_id = db.add_experiment("gaussian", "512x512", "abc", {"x": 1})
		assert exp_id is not None

		assert db.mark_profiled(exp_id, "kernel.ll")
		row = db.get_experiment_by_key("gaussian", "512x512", "abc")

	assert row is not None
	assert row.exp_id == exp_id
	assert row.status == "profiled"
	assert row.llvm_ir_path == "kernel.ll"
