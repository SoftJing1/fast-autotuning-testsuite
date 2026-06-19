from __future__ import annotations

import os
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
	build_generator_parameter_index_specs,
	build_parameter_index_specs,
	build_parameter_manipulator,
	config_from_parameter_indices,
	config_from_tuning_indices,
	indices_from_config,
)
from two_step_autotuning.core.parameter_resolver import DatabaseApproxParameterResolver
from two_step_autotuning.core.resolver import DatabaseApproxInstructionMapResolver
from two_step_autotuning.core.types import LiveProfile
from two_step_autotuning.live.config_space import LiveConfigGenerator
from scripts.internal.db_manager import ExperimentDB
from two_step_autotuning.live.mutation import LocalInstructionMapMutator, instruction_map_distance


DB_PATH = ROOT / os.environ.get(
	"TWO_STEP_AUTOTUNING_DB",
	"experiments/exp_20260415_large_scale_5000cfg/experiments.db",
)


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


def test_instruction_resolver_does_not_use_runtime_to_break_duplicate_map_ties():
	dataset = TuningDataset(DB_PATH, "gaussian", "512x512")
	duplicate_group = next(
		records
		for records in dataset.records_by_instruction_key.values()
		if len(records) > 1
	)
	assert duplicate_group[0].runtime_ms != min(record.runtime_ms for record in duplicate_group)

	resolver = DatabaseApproxInstructionMapResolver(dataset)
	result = resolver.resolve(duplicate_group[0].raw_counts)

	assert result.valid
	assert result.record is not None
	assert result.record.exp_id == duplicate_group[0].exp_id
	assert result.record.runtime_ms == duplicate_group[0].runtime_ms
	assert result.duplicate_count == len(duplicate_group)


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
		LiveProfile(
			1,
			"gaussian",
			"512x512",
			"a",
			{"x": 1},
			{"add": 10, "mul": 2, "lifetime.start": 100, "lifetime.end": 100},
		),
		LiveProfile(
			2,
			"gaussian",
			"512x512",
			"b",
			{"x": 2},
			{"add": 20, "mul": 2, "lifetime.start": 200, "lifetime.end": 200},
		),
		LiveProfile(
			3,
			"gaussian",
			"512x512",
			"c",
			{"x": 3},
			{"add": 30, "mul": 4, "lifetime.start": 300, "lifetime.end": 300},
		),
	]

	specs = build_instruction_parameter_specs_from_profiles(profiles, max_values_per_op=3)
	assert specs["add"].values == (10, 20, 30)
	assert specs["mul"].values == (2, 4)
	assert "lifetime.start" not in specs
	assert "lifetime.end" not in specs

	indices = nearest_indices_from_counts({"add": 21, "mul": 4}, specs)
	assert counts_from_tuning_indices(indices, specs) == {"add": 20, "mul": 4}
	assert median_instruction_counts(profiles) == {"add": 20, "mul": 2}


def test_live_generator_parameter_space_roundtrips_gaussian_config():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	specs = build_generator_parameter_index_specs("gaussian", "512x512", device_type="cpu")
	config = generator.generate(1)[0]

	indices = indices_from_config(config, specs)
	decoded = config_from_parameter_indices(indices, specs, fixed_config=generator.fixed_input_config())

	assert decoded == config
	assert indices


def test_live_generator_validation_rejects_invalid_gaussian_config():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	invalid_config = {
		"g_cb_res_dest_level": 2,
		"l_cb_res_dest_level": 0,
		"p_cb_res_dest_level": 0,
		"in_cache_lcl": 0,
		"in_cache_prv": 0,
		"out_cache_prv": 0,
		"wg_1_ocl_dim": 1,
		"wg_2_ocl_dim": 0,
		"wi_1_ocl_dim": 1,
		"wi_2_ocl_dim": 0,
		"input_size_1": 512,
		"input_size_2": 512,
		"input_size_h": 512,
		"input_size_w": 512,
		"glb_1": 3,
		"wg_1": 5,
		"lcl_1": 7,
		"wi_1": 11,
		"prv_1": 13,
		"glb_2": 3,
		"wg_2": 5,
		"lcl_2": 7,
		"wi_2": 11,
		"prv_2": 13,
	}

	assert not generator.validate_config(invalid_config)


def test_live_generator_generate_returns_valid_gemm_start_config():
	generator = LiveConfigGenerator("gemm", "128x128x128", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]

	assert generator.validate_config(config)


def test_experiment_db_can_store_profiled_rows_by_key(tmp_path):
	db_path = tmp_path / "live.db"
	with ExperimentDB(str(db_path)) as db:
		exp_id = db.add_experiment("gaussian", "512x512", "abc", {"x": 1})
		assert exp_id is not None

		assert db.mark_profiled(exp_id, "kernel.ll")
		db.add_experiment_log(
			exp_id,
			stage="runtime",
			level="warning",
			message="partial success",
			payload={"successful_runs": 2, "attempted_runs": 3},
		)
		row = db.get_experiment_by_key("gaussian", "512x512", "abc")
		logs = db.get_experiment_logs(exp_id)

	assert row is not None
	assert row.exp_id == exp_id
	assert row.status == "profiled"
	assert row.llvm_ir_path == "kernel.ll"
	assert len(logs) == 1
	assert logs[0]["stage"] == "runtime"
	assert logs[0]["payload"] == {"successful_runs": 2, "attempted_runs": 3}


def test_instruction_map_distance_ignores_excluded_ops():
	requested = {"add": 10, "mul": 4, "lifetime.start": 100}
	candidate = {"add": 13, "mul": 8, "lifetime.end": 200}

	assert instruction_map_distance(requested, candidate) == 5.0


def test_local_instruction_map_mutator_budget_zero_keeps_reference():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	executor = type("Executor", (), {})()
	mutator = LocalInstructionMapMutator(generator, executor, mutation_budget=0)
	config = generator.generate(1)[0]
	profile = LiveProfile(
		1,
		"gaussian",
		"512x512",
		"seed",
		config,
		{"add": 10, "mul": 4},
	)

	result = mutator.refine({"add": 10, "mul": 4}, profile)

	assert result.profile == profile
	assert result.distance == 0.0
	assert result.source == "database"


def test_local_instruction_map_mutator_can_pick_closer_neighbor():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]

	class FakeResult:
		def __init__(self, profile):
			self.profile = profile

		@property
		def valid(self):
			return self.profile is not None

	class FakeExecutor:
		def __init__(self, profiles_by_key):
			self.profiles_by_key = profiles_by_key

		def profile_config(self, config):
			return FakeResult(self.profiles_by_key.get(canonical_config_key(config)))

	reference_profile = LiveProfile(
		1,
		"gaussian",
		"512x512",
		"ref",
		config,
		{"add": 50, "mul": 20},
	)
	executor = FakeExecutor({})
	mutator = LocalInstructionMapMutator(generator, executor, mutation_budget=4)
	neighbors = mutator._neighbor_configs(
		indices_from_config(config, mutator.parameter_specs),
		{tuple(sorted(config.items()))},
	)
	assert neighbors
	neighbor_config, _ = neighbors[0]
	neighbor_profile = LiveProfile(
		2,
		"gaussian",
		"512x512",
		"neighbor",
		neighbor_config,
		{"add": 12, "mul": 5},
	)
	executor.profiles_by_key[canonical_config_key(neighbor_config)] = neighbor_profile

	result = mutator.refine({"add": 10, "mul": 4}, reference_profile)

	assert result.profile == neighbor_profile
	assert result.source == "mutation"
	assert result.distance < instruction_map_distance({"add": 10, "mul": 4}, reference_profile.raw_counts)


def test_local_instruction_map_mutator_caches_profiled_neighbors():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]

	class FakeResult:
		def __init__(self, profile):
			self.profile = profile

		@property
		def valid(self):
			return self.profile is not None

	class CountingExecutor:
		def __init__(self, profiles_by_key):
			self.profiles_by_key = profiles_by_key
			self.calls = 0

		def profile_config(self, config):
			self.calls += 1
			return FakeResult(self.profiles_by_key.get(canonical_config_key(config)))

	reference_profile = LiveProfile(
		1,
		"gaussian",
		"512x512",
		"ref",
		config,
		{"add": 50, "mul": 20},
	)
	executor = CountingExecutor({})
	mutator = LocalInstructionMapMutator(generator, executor, mutation_budget=4)
	neighbors = mutator._neighbor_configs(
		indices_from_config(config, mutator.parameter_specs),
		{tuple(sorted(config.items()))},
	)
	assert neighbors
	neighbor_config, _ = neighbors[0]
	neighbor_profile = LiveProfile(
		2,
		"gaussian",
		"512x512",
		"neighbor",
		neighbor_config,
		{"add": 12, "mul": 5},
	)
	executor.profiles_by_key[canonical_config_key(neighbor_config)] = neighbor_profile

	first_profile, first_cache_hit = mutator._profile_or_cache_hit(neighbor_config)
	second_profile, second_cache_hit = mutator._profile_or_cache_hit(neighbor_config)

	assert first_profile == neighbor_profile
	assert second_profile == neighbor_profile
	assert not first_cache_hit
	assert second_cache_hit
	assert executor.calls == 1
