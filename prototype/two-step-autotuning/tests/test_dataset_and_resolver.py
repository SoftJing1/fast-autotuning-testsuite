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
from two_step_autotuning.core.result_recorder import ResultRecorder, TRACE_FIELDS
from two_step_autotuning.core.types import LiveProfile
from two_step_autotuning.live.config_space import LiveConfigGenerator
from two_step_autotuning.live.two_step_tuner import (
	build_argparser as build_two_step_argparser,
	resolver_distance_ratio,
	resolver_mode_for_refinement_mode,
)
from scripts import collect_tuning_performance_simple as collector
from scripts.internal.db_manager import ExperimentDB
from two_step_autotuning.live.inner_parameter_tuner import (
	InnerParameterTuningInterface,
	instruction_map_distance,
)


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

	distances = model.component_distances_to_counts(dataset.records[10].raw_counts)

	assert distances[10] == 0.0


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
	assert len(specs["g_cb_res_dest_level"].values) == 3
	assert len(specs["wg_1_ocl_dim"].values) == 2
	assert len(specs["wi_1_ocl_dim"].values) == 2


def test_live_generator_validation_rejects_bad_gaussian_cache_hierarchy():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	config["g_cb_res_dest_level"] = 0
	config["l_cb_res_dest_level"] = 1
	config["p_cb_res_dest_level"] = 0

	assert not generator.validate_config(config)


def test_live_generator_validation_rejects_duplicate_gaussian_ocl_dims():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	config["wg_1_ocl_dim"] = 1
	config["wg_2_ocl_dim"] = 1

	assert not generator.validate_config(config)


def test_live_generator_validation_accepts_swapped_gaussian_ocl_dims():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	config["wg_1_ocl_dim"] = 0
	config["wg_2_ocl_dim"] = 1
	config["wi_1_ocl_dim"] = 0
	config["wi_2_ocl_dim"] = 1

	assert generator.validate_config(config)


def test_live_generator_validation_rejects_invalid_gaussian_config():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	invalid_config = {
		"g_cb_res_dest_level": 2,
		"l_cb_res_dest_level": 0,
		"p_cb_res_dest_level": 0,
		"images_cache_lcl": 0,
		"images_cache_prv": 0,
		"filter_cache_lcl": 0,
		"filter_cache_prv": 0,
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


def test_dataset_can_load_profiled_instruction_map_rows_without_runtime(tmp_path):
	db_path = tmp_path / "profiled.db"
	config = {
		"input_size_h": 512,
		"input_size_w": 512,
		"g_cb_res_dest_level": 2,
	}
	with ExperimentDB(str(db_path)) as db:
		exp_id = db.add_experiment("gaussian", "512x512", "abc", config)
		assert exp_id is not None
		db.upsert_llvm_instruction_counts(
			exp_id=exp_id,
			template_name="gaussian_static_1",
			kernel_function="gaussian_1",
			llvm_ir_path="transient://kernel.ll",
			bb_counts={},
			bb_instruction_counts={},
			total_instruction_counts={"add": 12, "mul": 3},
		)
		assert db.mark_profiled(exp_id)

	dataset = TuningDataset(db_path, "gaussian", "512x512", require_runtime=False)

	assert len(dataset.records) == 1
	assert dataset.records[0].runtime_ms is None
	assert dataset.records[0].raw_counts == {"add": 12, "mul": 3}


def test_dataset_runtime_required_skips_profiled_rows(tmp_path):
	db_path = tmp_path / "profiled.db"
	with ExperimentDB(str(db_path)) as db:
		exp_id = db.add_experiment("gaussian", "512x512", "abc", {"input_size_h": 512, "input_size_w": 512})
		assert exp_id is not None
		db.upsert_llvm_instruction_counts(
			exp_id=exp_id,
			template_name="gaussian_static_1",
			kernel_function="gaussian_1",
			llvm_ir_path="transient://kernel.ll",
			bb_counts={},
			bb_instruction_counts={},
			total_instruction_counts={"add": 12},
		)
		assert db.mark_profiled(exp_id)

	try:
		TuningDataset(db_path, "gaussian", "512x512")
	except ValueError as exc:
		assert "No completed records" in str(exc)
	else:
		raise AssertionError("runtime-required dataset should reject profiled-only rows")


def test_instruction_map_only_worker_marks_row_profiled_without_runtime(tmp_path, monkeypatch):
	db_path = tmp_path / "experiments.db"
	llvm_dir = tmp_path / "llvm"
	config_path = tmp_path / "config.json"
	config = {
		"input_size_h": 512,
		"input_size_w": 512,
		"g_cb_res_dest_level": 2,
	}
	config_path.write_text(__import__("json").dumps(config))
	param_hash = collector.hash_config(config)
	with ExperimentDB(str(db_path)) as db:
		exp_id = db.add_experiment("gaussian", "512x512", param_hash, config)
		assert exp_id is not None

	def fake_dump(kernel_type, config_file, build_dir, dump_opencl_binary):
		return collector.KernelRunResult(runtime_ms=0.0, returncode=0)

	def fake_extract(kernel_type, config_dict, param_hash_value, llvm_output_dir, opencl_binary_path, persist_artifacts):
		artifact_path = Path(llvm_output_dir) / "fake.ll"
		artifact_path.parent.mkdir(parents=True, exist_ok=True)
		artifact_path.write_text("; fake")
		return collector.RuntimeIRResult(
			artifacts=[
				{
					"template_name": "gaussian_static_1",
					"kernel_function": "gaussian_1",
					"llvm_ir_path": str(artifact_path),
					"llvm_ir_ref": "transient://fake.ll",
				}
			]
		)

	def fake_store(db, exp_id_value, llvm_artifacts):
		db.upsert_llvm_instruction_counts(
			exp_id=exp_id_value,
			template_name="gaussian_static_1",
			kernel_function="gaussian_1",
			llvm_ir_path="transient://fake.ll",
			bb_counts={},
			bb_instruction_counts={},
			total_instruction_counts={"add": 5},
		)
		return []

	monkeypatch.setattr(collector, "dump_opencl_binary_only", fake_dump)
	monkeypatch.setattr(collector, "_extract_runtime_ir_result", fake_extract)
	monkeypatch.setattr(collector, "_store_symbolic_instruction_counts", fake_store)

	result = collector.process_single_experiment(
		(
			exp_id,
			"gaussian",
			str(config_path),
			str(db_path),
			str(llvm_dir),
			"build",
			1,
			False,
			"instruction-map-only",
		)
	)

	with ExperimentDB(str(db_path)) as db:
		row = db.get_experiment_by_key("gaussian", "512x512", param_hash)
		counts = db.get_llvm_instruction_counts(exp_id)

	assert result == (exp_id, True, "")
	assert row is not None
	assert row.status == "profiled"
	assert row.runtime_ms is None
	assert counts[0]["total_instruction_counts"] == {"add": 5}


def test_instruction_map_distance_ignores_excluded_ops():
	requested = {"add": 10, "mul": 4, "lifetime.start": 100}
	candidate = {"add": 13, "mul": 8, "lifetime.end": 200}

	assert instruction_map_distance(requested, candidate) == 5.0


def test_normalized_instruction_map_distance_uses_scale_opcode_set():
	requested = {"add": 10}
	candidate = {"add": 14, "extra.live.op": 1000000}

	assert instruction_map_distance(requested, candidate, {"add": 2.0}) == 2.0


class FakeProfileResult:
	def __init__(self, profile):
		self.profile = profile

	@property
	def valid(self):
		return self.profile is not None


class FakeInnerExecutor:
	def __init__(self, profiles_by_key):
		self.profiles_by_key = profiles_by_key
		self.calls = []

	def profile_config(self, config):
		self.calls.append(config)
		return FakeProfileResult(self.profiles_by_key.get(canonical_config_key(config)))


def _inner_args(**overrides):
	from types import SimpleNamespace

	values = {
		"parallel_compile": False,
		"resolver_inner_valid_profile_limit": 32,
		"resolver_inner_test_limit": 512,
		"resolver_inner_target_ratio": 0.1,
		"resolver_inner_invalid_distance_penalty": 1.0e12,
	}
	values.update(overrides)
	return SimpleNamespace(**values)


def test_inner_parameter_tuner_seed_list_contains_only_db_config():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	profile = LiveProfile(1, "gaussian", "512x512", "seed", config, {"add": 10, "mul": 4})
	executor = FakeInnerExecutor({canonical_config_key(config): profile})
	tuner = InnerParameterTuningInterface(
		_inner_args(),
		generator,
		executor,
		{"add": 10, "mul": 4},
		profile,
		0.0,
	)

	seeds = tuner.seed_configurations()

	assert len(seeds) == 1
	assert seeds[0] == indices_from_config(config, tuner.parameter_specs)


def test_inner_parameter_tuner_invalid_decoded_config_returns_penalty():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	profile = LiveProfile(1, "gaussian", "512x512", "seed", config, {"add": 10, "mul": 4})
	executor = FakeInnerExecutor({canonical_config_key(config): profile})
	tuner = InnerParameterTuningInterface(
		_inner_args(resolver_inner_invalid_distance_penalty=12345.0),
		generator,
		executor,
		{"add": 10, "mul": 4},
		profile,
		0.0,
	)
	index_config = indices_from_config(config, tuner.parameter_specs)
	index_config["idx.g_cb_res_dest_level"] = 0
	index_config["idx.l_cb_res_dest_level"] = 1
	index_config["idx.p_cb_res_dest_level"] = 0
	from types import SimpleNamespace

	result = tuner.run(SimpleNamespace(configuration=SimpleNamespace(data=index_config)))

	assert result.time == 12345.0
	assert tuner.invalid_count == 1
	assert tuner.best_profile is None


def test_inner_parameter_tuner_valid_profile_distance_is_result_time():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	profile = LiveProfile(1, "gaussian", "512x512", "seed", config, {"add": 13, "mul": 8})
	executor = FakeInnerExecutor({canonical_config_key(config): profile})
	tuner = InnerParameterTuningInterface(
		_inner_args(),
		generator,
		executor,
		{"add": 10, "mul": 4},
		profile,
		5.0,
	)
	from types import SimpleNamespace

	result = tuner.run(
		SimpleNamespace(configuration=SimpleNamespace(data=indices_from_config(config, tuner.parameter_specs)))
	)

	assert result.time == 5.0
	assert tuner.best_profile == profile
	assert tuner.best_distance == 5.0


def test_inner_parameter_tuner_target_hit_stops_after_seed():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	profile = LiveProfile(1, "gaussian", "512x512", "seed", config, {"add": 10, "mul": 4})
	executor = FakeInnerExecutor({canonical_config_key(config): profile})
	tuner = InnerParameterTuningInterface(
		_inner_args(resolver_inner_target_ratio=0.1),
		generator,
		executor,
		{"add": 10, "mul": 4},
		profile,
		10.0,
	)

	result = tuner.tune()

	assert result.target_hit
	assert result.total_tests == 1
	assert result.valid_profile_count == 1


def test_inner_parameter_tuner_budget_exhaustion_returns_best_valid_profile():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	profile = LiveProfile(1, "gaussian", "512x512", "seed", config, {"add": 13, "mul": 8})
	executor = FakeInnerExecutor({canonical_config_key(config): profile})
	tuner = InnerParameterTuningInterface(
		_inner_args(resolver_inner_test_limit=1, resolver_inner_target_ratio=0.0),
		generator,
		executor,
		{"add": 10, "mul": 4},
		profile,
		5.0,
	)

	result = tuner.tune()

	assert result.profile == profile
	assert result.distance == 5.0
	assert result.total_tests == 1


def test_inner_parameter_tuner_zero_db_distance_ratio_is_safe():
	generator = LiveConfigGenerator("gaussian", "512x512", device_type="cpu", random_seed=1)
	config = generator.generate(1)[0]
	profile = LiveProfile(1, "gaussian", "512x512", "seed", config, {"add": 10, "mul": 4})
	executor = FakeInnerExecutor({canonical_config_key(config): profile})
	tuner = InnerParameterTuningInterface(
		_inner_args(),
		generator,
		executor,
		{"add": 10, "mul": 4},
		profile,
		0.0,
	)

	result = tuner.tune()

	assert result.distance_ratio == 1.0
	assert result.target_distance == 0.0
	assert result.target_hit


def test_resolver_distance_ratio_handles_zero_and_missing_values():
	assert resolver_distance_ratio(10.0, 2.5) == 0.25
	assert resolver_distance_ratio(0.0, 0.0) == 1.0
	assert resolver_distance_ratio(None, 1.0) is None
	assert resolver_distance_ratio(1.0, None) is None
	assert resolver_distance_ratio(0.0, 1.0) is None


def test_resolver_mode_reflects_refinement_mode():
	assert resolver_mode_for_refinement_mode("database") == "database_only"
	assert resolver_mode_for_refinement_mode("inner-tuner") == "database_plus_inner_tuner"


def test_two_step_help_no_longer_exposes_mutation_flags():
	help_text = build_two_step_argparser().format_help()

	assert "resolver-mutation-budget" not in help_text
	assert "mutation" not in help_text
	assert "resolver-refinement-mode" in help_text


def test_result_recorder_accepts_resolver_diagnostic_fields(tmp_path):
	assert "db_resolver_distance" in TRACE_FIELDS
	assert "refined_resolver_distance" in TRACE_FIELDS
	assert "distance_ratio" in TRACE_FIELDS
	assert "resolver_source" in TRACE_FIELDS
	assert "inner_valid_profile_count" in TRACE_FIELDS

	recorder = ResultRecorder(tmp_path, {"method": "test"})
	recorder.record(
		{
			"candidate_status": "inner_tuned_valid",
			"runtime_ms": 1.25,
			"exp_id": 7,
			"param_hash": "resolved",
			"resolver_distance": 2.0,
			"db_resolver_distance": 20.0,
			"refined_resolver_distance": 2.0,
			"distance_ratio": 0.1,
			"resolver_source": "inner_tuner",
			"db_param_hash": "db",
			"resolved_param_hash": "resolved",
			"resolver_profile_count": 4,
			"resolver_time_ms": 0.5,
			"resolver_refinement_time_ms": 12.0,
			"duplicate_count": 1,
			"inner_valid_profile_count": 4,
			"inner_total_tests": 9,
			"inner_invalid_count": 5,
			"inner_target_distance": 2.0,
			"inner_target_hit": True,
			"inner_wall_time_ms": 12.0,
		}
	)

	trace = (tmp_path / "trace.csv").read_text()
	assert "db_resolver_distance" in trace
	assert "20.0" in trace
	assert "inner_tuned_valid" in trace
	assert "inner_valid_profile_count" in trace
