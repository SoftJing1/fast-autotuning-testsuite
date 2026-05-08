#!/usr/bin/env python3
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from scripts.internal.visualize_results_web import (  # noqa: E402
	ALL_TUNING_PARAM_VALUES,
	_apply_tuning_param_filters,
	_build_heatmap_matrix,
	_build_histogram_figure,
	_build_launch_geometry,
	_filtered_counts,
	_load_instruction_mix_dataset,
	_overlay_records,
	_shortcut_param_target,
	_tuning_param_names,
	_tuning_param_shortcut_options,
)


GAUSSIAN_CONFIG = {
	"wi_1_ocl_dim": 1,
	"wi_2_ocl_dim": 0,
	"wi_1": 1,
	"wi_2": 16,
	"wg_1_ocl_dim": 1,
	"wg_2_ocl_dim": 0,
	"wg_1": 4,
	"wg_2": 8,
}


GEMM_CONFIG = {
	"ocl_dim_l_1": 2,
	"num_wg_l_1": 32,
	"num_wi_l_1": 1,
	"ocl_dim_l_2": 0,
	"num_wg_l_2": 32,
	"num_wi_l_2": 1,
	"ocl_dim_r_1": 1,
	"num_wg_r_1": 64,
	"num_wi_r_1": 1,
}


def _create_fixture_db(db_path: Path) -> None:
	conn = sqlite3.connect(str(db_path))
	try:
		conn.executescript(
			"""
			CREATE TABLE experiments (
				exp_id INTEGER PRIMARY KEY,
				kernel_type TEXT NOT NULL,
				input_size TEXT NOT NULL,
				param_hash TEXT NOT NULL,
				config_json TEXT NOT NULL,
				runtime_ms REAL,
				llvm_ir_path TEXT,
				timestamp TEXT NOT NULL,
				status TEXT NOT NULL
			);
			CREATE TABLE llvm_instruction_counts (
				id INTEGER PRIMARY KEY,
				exp_id INTEGER NOT NULL,
				template_name TEXT NOT NULL,
				kernel_function TEXT NOT NULL,
				llvm_ir_path TEXT NOT NULL,
				bb_counts_json TEXT NOT NULL,
				bb_instruction_counts_json TEXT NOT NULL,
				total_instruction_counts_json TEXT NOT NULL,
				created_at TEXT NOT NULL
			);
			"""
		)
		conn.execute(
			"""
			INSERT INTO experiments
			(exp_id, kernel_type, input_size, param_hash, config_json, runtime_ms, llvm_ir_path, timestamp, status)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
			(1, "gaussian", "512x512", "gauss_hash", json.dumps(GAUSSIAN_CONFIG), 1.5, "", "now", "completed"),
		)
		conn.execute(
			"""
			INSERT INTO experiments
			(exp_id, kernel_type, input_size, param_hash, config_json, runtime_ms, llvm_ir_path, timestamp, status)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
			(2, "gemm", "128x128x128", "gemm_hash", json.dumps(GEMM_CONFIG), 2.5, "", "now", "completed"),
		)
		for row in [
			(1, "gaussian_static_1", "gaussian_1", {"load": 2, "call": 3}),
			(2, "gemm_1", "gemm_1", {"load": 2, "mul": 1, "call": 1}),
			(2, "gemm_2", "gemm_2", {"load": 5, "store": 2}),
		]:
			exp_id, template_name, kernel_function, counts = row
			conn.execute(
				"""
				INSERT INTO llvm_instruction_counts
				(exp_id, template_name, kernel_function, llvm_ir_path, bb_counts_json, bb_instruction_counts_json,
				 total_instruction_counts_json, created_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				""",
				(exp_id, template_name, kernel_function, "", "{}", "{}", json.dumps(counts), "now"),
			)
		conn.commit()
	finally:
		conn.close()


def test_launch_thread_count_matches_existing_geometry_formulas():
	gaussian = _build_launch_geometry("gaussian", "gaussian_static_1", GAUSSIAN_CONFIG)
	assert gaussian.local_sizes == (16, 1, 1)
	assert gaussian.num_groups == (8, 4, 1)
	assert gaussian.thread_count == 512

	gemm_1 = _build_launch_geometry("gemm", "gemm_1", GEMM_CONFIG)
	assert gemm_1.local_sizes == (1, 1, 1)
	assert gemm_1.num_groups == (32, 64, 32)
	assert gemm_1.thread_count == 65_536

	gemm_2 = _build_launch_geometry("gemm", "gemm_2", GEMM_CONFIG)
	assert gemm_2.local_sizes == (1, 1, 1)
	assert gemm_2.num_groups == (32, 1, 32)
	assert gemm_2.thread_count == 1_024


def test_instruction_mix_dataset_scales_and_combines_counts(tmp_path):
	db_path = tmp_path / "experiments.db"
	_create_fixture_db(db_path)
	_load_instruction_mix_dataset.cache_clear()

	dataset = _load_instruction_mix_dataset(str(db_path))
	records = dataset["records"]
	gaussian = next(record for record in records if record["kernel_type"] == "gaussian")
	assert gaussian["variant"] == "primary"
	assert gaussian["scaled_counts"]["load"] == 1_024
	assert gaussian["scaled_counts"]["call"] == 1_536
	assert _filtered_counts(gaussian, include_call=False) == {"load": 1_024}

	gemm_primary = next(record for record in records if record["kernel_type"] == "gemm" and record["variant"] == "primary")
	assert gemm_primary["scaled_counts"]["load"] == 131_072
	assert gemm_primary["scaled_counts"]["call"] == 65_536

	gemm_combined = next(record for record in records if record["kernel_type"] == "gemm" and record["variant"] == "combined")
	assert gemm_combined["scaled_counts"]["load"] == 136_192
	assert gemm_combined["scaled_counts"]["store"] == 2_048
	assert gemm_combined["scaled_counts"]["call"] == 65_536


def test_heatmap_matrix_uses_row_mix_then_total_darkening():
	records = [
		{"scaled_counts": {"load": 10, "store": 5}},
		{"scaled_counts": {"load": 100, "store": 0}},
	]
	matrix = _build_heatmap_matrix(records, records, ["load", "store"], include_call=True)

	assert matrix[0][0] == pytest.approx(0.25)
	assert matrix[0][1] == pytest.approx(0.125)
	assert matrix[1][0] == pytest.approx(1.0)
	assert matrix[1][1] == pytest.approx(0.0)


def test_heatmap_matrix_can_use_portion_only_color():
	records = [
		{"scaled_counts": {"load": 10, "store": 5}},
		{"scaled_counts": {"load": 100, "store": 0}},
	]
	matrix = _build_heatmap_matrix(records, records, ["load", "store"], include_call=True, color_mode="portion_only")

	assert matrix[0][0] == pytest.approx(10 / 15)
	assert matrix[0][1] == pytest.approx(5 / 15)
	assert matrix[1][0] == pytest.approx(1.0)
	assert matrix[1][1] == pytest.approx(0.0)


def test_histogram_figure_skips_all_zero_opcodes():
	records = [
		{"scaled_counts": {"load": 10, "store": 0}},
		{"scaled_counts": {"load": 20, "store": 0}},
	]
	figure = _build_histogram_figure(records, ["load", "store"], include_call=True, bin_count_value=4)

	assert len(figure.data) == 1
	assert figure.layout.annotations[0].text == "load"


def test_histogram_figure_overlays_performance_group():
	records = [
		{"scaled_counts": {"load": 10}},
		{"scaled_counts": {"load": 20}},
		{"scaled_counts": {"load": 30}},
	]
	overlay = [records[0], records[2]]
	figure = _build_histogram_figure(records, ["load"], include_call=True, bin_count_value=4, overlay_records=overlay, overlay_label="Fastest kernels")

	assert len(figure.data) == 2
	assert figure.data[0].name == "Current selection"
	assert figure.data[1].name == "Fastest kernels"
	assert sum(figure.data[1].y) == 2
	assert figure.layout.barmode == "overlay"


def test_overlay_records_selects_requested_performance_group():
	records = [
		{"exp_id": 1, "runtime_ms": 30.0, "scaled_counts": {"load": 100}},
		{"exp_id": 2, "runtime_ms": 10.0, "scaled_counts": {"load": 300}},
		{"exp_id": 3, "runtime_ms": 20.0, "scaled_counts": {"load": 200}},
	]

	fastest, fastest_limit = _overlay_records(records, "fastest", 2, include_call=True)
	most_instructions, total_limit = _overlay_records(records, "largest_total", 1, include_call=True)

	assert fastest_limit == 2
	assert [record["exp_id"] for record in fastest] == [2, 3]
	assert total_limit == 1
	assert [record["exp_id"] for record in most_instructions] == [2]


def test_tuning_param_filters_keep_selected_values_only():
	records = [
		{"config": {"wg_1": 4, "wi_1": 1}, "scaled_counts": {"load": 10}},
		{"config": {"wg_1": 8, "wi_1": 1}, "scaled_counts": {"load": 20}},
		{"config": {"wg_1": 8, "wi_1": 2}, "scaled_counts": {"load": 30}},
	]
	filter_ids = [
		{"type": "mix-heatmap-param-filter", "param": "wg_1"},
		{"type": "mix-heatmap-param-filter", "param": "wi_1"},
	]
	filter_values = ["8", ALL_TUNING_PARAM_VALUES]

	filtered = _apply_tuning_param_filters(records, filter_ids, filter_values)

	assert [record["scaled_counts"]["load"] for record in filtered] == [20, 30]


def test_tuning_param_shortcut_options_keep_raw_param_names_and_all_default():
	records = [
		{"config": {"wg_1": 4, "wi_1": 1, "input_size_h": 512}},
		{"config": {"wg_1": 8, "wi_1": 2, "input_size_h": 512}},
	]

	assert _tuning_param_names(records) == ["wg_1", "wi_1"]
	assert _tuning_param_shortcut_options(records, "wg_1") == [ALL_TUNING_PARAM_VALUES, "4", "8"]
	assert _shortcut_param_target('{"param":"wg_1","type":"mix-heatmap-param-filter"}') == {
		"param": "wg_1",
		"type": "mix-heatmap-param-filter",
	}
