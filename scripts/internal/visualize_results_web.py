#!/usr/bin/env python3
"""Web-based interactive visualization for experiment SQLite databases.

Run this script against an experiment database (experiments.db) to inspect
summary metrics, filterable results, and runtime trends.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import socket
import sqlite3
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
	from dash import ALL, Dash, Input, Output, State, callback, clientside_callback, ctx, dash_table, dcc, html, no_update
	import plotly.graph_objects as go
	from plotly.subplots import make_subplots
except ImportError as exc:
	DASH_IMPORT_ERROR = exc
	ALL = Dash = Input = Output = State = callback = clientside_callback = ctx = dash_table = dcc = html = no_update = None
	go = make_subplots = None
else:
	DASH_IMPORT_ERROR = None


DEEP_DIVE_RELATIVE_DIR = Path("instruction_mix_deep_dive") / "full_interactive"
MAX_HEATMAP_ROWS = 1000
MAX_HISTOGRAM_ROWS = 10000
ALL_TUNING_PARAM_VALUES = "__all__"
INPUT_SIZE_CONFIG_KEYS = {
	"input_size_h",
	"input_size_w",
	"input_size_1",
	"input_size_2",
	"input_size_l_1",
	"input_size_l_2",
	"input_size_r_1",
	"M",
	"N",
	"K",
}


@dataclass(frozen=True)
class LaunchGeometry:
	local_sizes: tuple[int, int, int]
	num_groups: tuple[int, int, int]

	@property
	def thread_count(self) -> int:
		return _product(self.local_sizes) * _product(self.num_groups)

	def label(self) -> str:
		return f"local={self.local_sizes}, groups={self.num_groups}, threads={self.thread_count}"


def _query_rows(db_path: Path, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(sql, params).fetchall()
		return [dict(row) for row in rows]
	finally:
		conn.close()


def _query_one(db_path: Path, sql: str, params: tuple = ()) -> Dict[str, Any]:
	rows = _query_rows(db_path, sql, params)
	return rows[0] if rows else {}


def _product(values: Iterable[int]) -> int:
	result = 1
	for value in values:
		result *= int(value)
	return result


def _to_int(value: Any, default: int = 0) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return default


def _format_count(value: Any) -> str:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return str(value)
	if abs(number) >= 1_000_000:
		return f"{number:.3e}"
	if number == int(number):
		return str(int(number))
	return f"{number:.4f}"


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
	if not sorted_values:
		return 0.0
	if len(sorted_values) == 1:
		return float(sorted_values[0])
	position = (len(sorted_values) - 1) * percentile
	lower = int(math.floor(position))
	upper = int(math.ceil(position))
	if lower == upper:
		return float(sorted_values[lower])
	weight = position - lower
	return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _mean(values: Sequence[float]) -> float:
	return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
	if len(values) < 2:
		return 0.0
	avg = _mean(values)
	return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def _build_launch_geometry(kernel_type: str, template_name: str, config: Dict[str, Any]) -> LaunchGeometry:
	if kernel_type == "gaussian":
		local_x = int((config["wi_1_ocl_dim"] == 0) * config["wi_1"] + (config["wi_2_ocl_dim"] == 0) * config["wi_2"])
		local_y = int((config["wi_1_ocl_dim"] == 1) * config["wi_1"] + (config["wi_2_ocl_dim"] == 1) * config["wi_2"])
		global_x = int(((config["wg_1_ocl_dim"] == 0) * config["wg_1"] + (config["wg_2_ocl_dim"] == 0) * config["wg_2"]) * local_x)
		global_y = int(((config["wg_1_ocl_dim"] == 1) * config["wg_1"] + (config["wg_2_ocl_dim"] == 1) * config["wg_2"]) * local_y)
		local = (max(local_x, 1), max(local_y, 1), 1)
		return LaunchGeometry(
			local_sizes=local,
			num_groups=(max(global_x // local[0], 1), max(global_y // local[1], 1), 1),
		)

	if kernel_type != "gemm":
		raise ValueError(f"Unsupported kernel type: {kernel_type}")

	is_reduction = template_name.endswith("gemm_2") or template_name == "gemm_2"
	local_sizes = [0, 0, 0]
	global_sizes = [0, 0, 0]
	dim_specs = [
		("l_1", config["ocl_dim_l_1"], config["num_wg_l_1"], config["num_wi_l_1"]),
		("l_2", config["ocl_dim_l_2"], config["num_wg_l_2"], config["num_wi_l_2"]),
		("r_1", config["ocl_dim_r_1"], config["num_wg_r_1"], config["num_wi_r_1"]),
	]
	for dim_name, ocl_dim, num_wg, num_wi in dim_specs:
		ocl_dim = int(ocl_dim)
		num_wi = int(num_wi)
		local_sizes[ocl_dim] += num_wi
		effective_num_wg = 1 if (is_reduction and dim_name == "r_1") else int(num_wg)
		global_sizes[ocl_dim] += effective_num_wg * num_wi

	for index in range(3):
		local_sizes[index] = max(local_sizes[index], 1)
		global_sizes[index] = max(global_sizes[index], local_sizes[index])

	return LaunchGeometry(
		local_sizes=tuple(local_sizes),
		num_groups=tuple(max(global_sizes[index] // local_sizes[index], 1) for index in range(3)),
	)


def _scale_instruction_counts(counts: Dict[str, Any], launch_threads: int) -> Dict[str, int]:
	return {op: _to_int(value) * int(launch_threads) for op, value in counts.items()}


def _sum_counts(count_sets: Iterable[Dict[str, int]]) -> Dict[str, int]:
	total: Dict[str, int] = defaultdict(int)
	for counts in count_sets:
		for op, value in counts.items():
			total[op] += int(value)
	return dict(total)


def _filtered_counts(record: Dict[str, Any], include_call: bool) -> Dict[str, int]:
	counts = record["scaled_counts"]
	if include_call:
		return dict(counts)
	return {op: value for op, value in counts.items() if op != "call"}


def _filtered_raw_counts(record: Dict[str, Any], include_call: bool) -> Dict[str, int]:
	counts = record["raw_counts"]
	if include_call:
		return dict(counts)
	return {op: value for op, value in counts.items() if op != "call"}


def _total_for_record(record: Dict[str, Any], include_call: bool) -> int:
	return int(sum(_filtered_counts(record, include_call).values()))


def _variant_label(variant: str) -> str:
	return {
		"primary": "Primary kernel",
		"combined": "GEMM combined",
		"gemm_2": "GEMM reduction kernel",
	}.get(variant, variant)


def _variant_sort_key(variant: str) -> tuple[int, str]:
	order = {"primary": 0, "combined": 1, "gemm_2": 2}
	return (order.get(variant, 99), variant)


FOCUS_OPTIONS = ["all", "long_runtime", "short_runtime", "large_total", "small_total"]
ROW_SORT_OPTIONS = ["exp_id", "runtime_desc", "runtime_asc", "scaled_desc", "scaled_asc"]
OPCODE_SORT_OPTIONS = ["name", "total_desc", "variance_desc"]
HEATMAP_COLOR_MODE_OPTIONS = ["mix_plus_total", "portion_only"]
HISTOGRAM_OVERLAY_OPTIONS = ["none", "fastest", "slowest", "largest_total", "smallest_total"]


def _cycle_option(options: Sequence[str], current_value: str | None, direction: int) -> str | None:
	if not options:
		return current_value
	if current_value not in options:
		return options[0]
	index = options.index(current_value)
	return options[(index + direction) % len(options)]


def _number_option(options: Sequence[str], key: str, current_value: str | None) -> str | None:
	try:
		index = int(key) - 1
	except (TypeError, ValueError):
		return current_value
	if 0 <= index < len(options):
		return options[index]
	return current_value


def _step_numeric(value: Any, direction: int, minimum: int, maximum: int, step: int) -> int:
	try:
		current = int(value)
	except (TypeError, ValueError):
		current = minimum
	return max(minimum, min(maximum, current + direction * step))


def _format_param_value(value: Any) -> str:
	if isinstance(value, bool):
		return "1" if value else "0"
	if isinstance(value, float) and value.is_integer():
		return str(int(value))
	return str(value)


def _param_sort_key(value: str) -> tuple[int, float | str]:
	try:
		return (0, float(value))
	except (TypeError, ValueError):
		return (1, value)


def _tuning_param_names(records: Sequence[Dict[str, Any]]) -> List[str]:
	params: set[str] = set()
	for record in records:
		params.update(record.get("config", {}).keys())
	return sorted(param for param in params if param not in INPUT_SIZE_CONFIG_KEYS)


def _tuning_param_values(records: Sequence[Dict[str, Any]], param: str) -> List[str]:
	values = {
		_format_param_value(record.get("config", {}).get(param))
		for record in records
		if param in record.get("config", {})
	}
	return sorted(values, key=_param_sort_key)


def _tuning_param_shortcut_options(records: Sequence[Dict[str, Any]], param: str) -> List[str]:
	return [ALL_TUNING_PARAM_VALUES] + _tuning_param_values(records, param)


def _shortcut_param_target(target: Any) -> Dict[str, Any] | None:
	if not isinstance(target, str):
		return None
	try:
		decoded = json.loads(target)
	except json.JSONDecodeError:
		return None
	if not isinstance(decoded, dict):
		return None
	if decoded.get("type") not in {"mix-heatmap-param-filter", "mix-hist-param-filter"}:
		return None
	if not decoded.get("param"):
		return None
	return decoded


def _apply_tuning_param_filters(
	records: List[Dict[str, Any]],
	filter_ids: Sequence[Dict[str, Any]] | None,
	filter_values: Sequence[Any] | None,
) -> List[Dict[str, Any]]:
	if not filter_ids or not filter_values:
		return records
	selected_by_param: Dict[str, str] = {}
	for filter_id, selected_value in zip(filter_ids, filter_values):
		param = str(filter_id.get("param", ""))
		if not param or selected_value in (None, ALL_TUNING_PARAM_VALUES):
			continue
		selected_by_param[param] = str(selected_value)
	if not selected_by_param:
		return records

	filtered = []
	for record in records:
		config = record.get("config", {})
		matches = True
		for param, selected_value in selected_by_param.items():
			if _format_param_value(config.get(param)) != selected_value:
				matches = False
				break
		if matches:
			filtered.append(record)
	return filtered


def _template_variant(kernel_type: str, template_name: str) -> str | None:
	if kernel_type == "gaussian":
		return "primary"
	if kernel_type == "gemm" and template_name == "gemm_1":
		return "primary"
	if kernel_type == "gemm" and template_name == "gemm_2":
		return "gemm_2"
	return None


@lru_cache(maxsize=8)
def _load_instruction_mix_dataset(db_path_str: str) -> Dict[str, Any]:
	db_path = Path(db_path_str)
	sql = """
		SELECT e.exp_id, e.kernel_type, e.input_size, e.param_hash, e.config_json, e.runtime_ms,
		       lic.template_name, lic.kernel_function, lic.total_instruction_counts_json
		FROM llvm_instruction_counts lic
		JOIN experiments e ON e.exp_id = lic.exp_id
		WHERE e.status = 'completed'
		ORDER BY e.kernel_type, e.input_size, e.exp_id, lic.template_name
	"""
	rows = _query_rows(db_path, sql)

	template_counts: Dict[tuple[str, str, str], int] = defaultdict(int)
	records: List[Dict[str, Any]] = []
	grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
	opcodes: set[str] = set()

	for row in rows:
		config = json.loads(row["config_json"])
		raw_counts = {str(op): _to_int(value) for op, value in json.loads(row["total_instruction_counts_json"]).items()}
		geometry = _build_launch_geometry(row["kernel_type"], row["template_name"], config)
		scaled_counts = _scale_instruction_counts(raw_counts, geometry.thread_count)
		opcodes.update(raw_counts)
		template_counts[(row["kernel_type"], row["input_size"], row["template_name"])] += 1
		template_row = {
			"exp_id": int(row["exp_id"]),
			"kernel_type": row["kernel_type"],
			"input_size": row["input_size"],
			"param_hash": row["param_hash"],
			"runtime_ms": row["runtime_ms"],
			"config_json": row["config_json"],
			"config": config,
			"template_name": row["template_name"],
			"kernel_function": row["kernel_function"],
			"raw_counts": raw_counts,
			"scaled_counts": scaled_counts,
			"launch_threads": geometry.thread_count,
			"launch_description": geometry.label(),
			"local_sizes": geometry.local_sizes,
			"num_groups": geometry.num_groups,
		}
		grouped[int(row["exp_id"])].append(template_row)

	for template_rows in grouped.values():
		for template_row in template_rows:
			variant = _template_variant(template_row["kernel_type"], template_row["template_name"])
			if variant is None:
				continue
			records.append(_make_mix_record(template_row, variant, _variant_label(variant)))

		gemm_rows = {row["template_name"]: row for row in template_rows if row["kernel_type"] == "gemm"}
		if "gemm_1" in gemm_rows and "gemm_2" in gemm_rows:
			primary = gemm_rows["gemm_1"]
			reduction = gemm_rows["gemm_2"]
			raw_counts = _sum_counts([primary["raw_counts"], reduction["raw_counts"]])
			scaled_counts = _sum_counts([primary["scaled_counts"], reduction["scaled_counts"]])
			records.append(
				{
					"exp_id": primary["exp_id"],
					"kernel_type": primary["kernel_type"],
					"input_size": primary["input_size"],
					"variant": "combined",
					"variant_label": _variant_label("combined"),
					"template_name": "gemm_1+gemm_2",
					"kernel_function": "gemm_1+gemm_2",
					"param_hash": primary["param_hash"],
					"runtime_ms": primary["runtime_ms"],
					"config_json": primary["config_json"],
					"config": dict(primary["config"]),
					"raw_counts": raw_counts,
					"scaled_counts": scaled_counts,
					"launch_threads": primary["launch_threads"] + reduction["launch_threads"],
					"launch_description": f"gemm_1: {primary['launch_description']} | gemm_2: {reduction['launch_description']}",
					"local_sizes": {"gemm_1": primary["local_sizes"], "gemm_2": reduction["local_sizes"]},
					"num_groups": {"gemm_1": primary["num_groups"], "gemm_2": reduction["num_groups"]},
				}
			)

	kernels = sorted({record["kernel_type"] for record in records})
	sizes_by_kernel = {
		kernel: sorted({record["input_size"] for record in records if record["kernel_type"] == kernel})
		for kernel in kernels
	}
	variants_by_kernel = {
		kernel: sorted({record["variant"] for record in records if record["kernel_type"] == kernel}, key=_variant_sort_key)
		for kernel in kernels
	}
	group_counts: Dict[str, int] = defaultdict(int)
	for record in records:
		key = _group_key(record["kernel_type"], record["input_size"], record["variant"])
		group_counts[key] += 1

	return {
		"records": records,
		"opcodes": sorted(opcodes),
		"kernels": kernels,
		"sizes_by_kernel": sizes_by_kernel,
		"variants_by_kernel": variants_by_kernel,
		"group_counts": dict(group_counts),
		"template_counts": [
			{
				"kernel_type": kernel,
				"input_size": input_size,
				"template_name": template_name,
				"count": count,
			}
			for (kernel, input_size, template_name), count in sorted(template_counts.items())
		],
	}


def _make_mix_record(template_row: Dict[str, Any], variant: str, variant_label: str) -> Dict[str, Any]:
	return {
		"exp_id": template_row["exp_id"],
		"kernel_type": template_row["kernel_type"],
		"input_size": template_row["input_size"],
		"variant": variant,
		"variant_label": variant_label,
		"template_name": template_row["template_name"],
		"kernel_function": template_row["kernel_function"],
		"param_hash": template_row["param_hash"],
		"runtime_ms": template_row["runtime_ms"],
		"config_json": template_row["config_json"],
		"config": dict(template_row["config"]),
		"raw_counts": dict(template_row["raw_counts"]),
		"scaled_counts": dict(template_row["scaled_counts"]),
		"launch_threads": template_row["launch_threads"],
		"launch_description": template_row["launch_description"],
		"local_sizes": template_row["local_sizes"],
		"num_groups": template_row["num_groups"],
	}


def _group_key(kernel_type: str, input_size: str, variant: str) -> str:
	return f"{kernel_type}:{input_size}:{variant}"


def _default_mix_selection(dataset: Dict[str, Any]) -> tuple[str | None, str | None, str | None]:
	kernel = "gaussian" if "gaussian" in dataset["kernels"] else (dataset["kernels"][0] if dataset["kernels"] else None)
	size = None
	variant = None
	if kernel:
		sizes = dataset["sizes_by_kernel"].get(kernel, [])
		size = sizes[0] if sizes else None
		variants = dataset["variants_by_kernel"].get(kernel, [])
		variant = "primary" if "primary" in variants else (variants[0] if variants else None)
	return kernel, size, variant


def _records_for_group(
	dataset: Dict[str, Any],
	kernel_type: str | None,
	input_size: str | None,
	variant: str | None,
) -> List[Dict[str, Any]]:
	if not kernel_type or not input_size or not variant:
		return []
	return [
		record
		for record in dataset["records"]
		if record["kernel_type"] == kernel_type
		and record["input_size"] == input_size
		and record["variant"] == variant
	]


def _sort_records(records: List[Dict[str, Any]], sort_mode: str, include_call: bool) -> List[Dict[str, Any]]:
	if sort_mode == "runtime_desc":
		return sorted(records, key=lambda record: float(record["runtime_ms"] or 0.0), reverse=True)
	if sort_mode == "runtime_asc":
		return sorted(records, key=lambda record: float(record["runtime_ms"] or 0.0))
	if sort_mode == "scaled_desc":
		return sorted(records, key=lambda record: _total_for_record(record, include_call), reverse=True)
	if sort_mode == "scaled_asc":
		return sorted(records, key=lambda record: _total_for_record(record, include_call))
	return sorted(records, key=lambda record: int(record["exp_id"]))


def _focused_records(
	records: List[Dict[str, Any]],
	focus_mode: str,
	limit_value: Any,
	row_sort: str,
	include_call: bool,
	max_limit: int = MAX_HEATMAP_ROWS,
) -> tuple[List[Dict[str, Any]], int]:
	try:
		limit = max(1, min(max_limit, int(limit_value or 300)))
	except (TypeError, ValueError):
		limit = min(300, max_limit)

	if focus_mode == "long_runtime":
		focused = _sort_records(records, "runtime_desc", include_call)[:limit]
	elif focus_mode == "short_runtime":
		focused = _sort_records(records, "runtime_asc", include_call)[:limit]
	elif focus_mode == "large_total":
		focused = _sort_records(records, "scaled_desc", include_call)[:limit]
	elif focus_mode == "small_total":
		focused = _sort_records(records, "scaled_asc", include_call)[:limit]
	else:
		focused = list(records)

	return _sort_records(focused, row_sort, include_call)[:limit], limit


def _overlay_records(
	records: List[Dict[str, Any]],
	overlay_mode: str | None,
	count_value: Any,
	include_call: bool,
) -> tuple[List[Dict[str, Any]], int]:
	if not records or not overlay_mode or overlay_mode == "none":
		return [], 0
	try:
		limit = max(1, min(MAX_HISTOGRAM_ROWS, int(count_value or 100)))
	except (TypeError, ValueError):
		limit = 100

	if overlay_mode == "fastest":
		selected = _sort_records(records, "runtime_asc", include_call)
	elif overlay_mode == "slowest":
		selected = _sort_records(records, "runtime_desc", include_call)
	elif overlay_mode == "largest_total":
		selected = _sort_records(records, "scaled_desc", include_call)
	elif overlay_mode == "smallest_total":
		selected = _sort_records(records, "scaled_asc", include_call)
	else:
		return [], limit
	return selected[:limit], limit


def _overlay_label(overlay_mode: str | None) -> str:
	return {
		"none": "No overlay",
		"fastest": "Fastest kernels",
		"slowest": "Slowest kernels",
		"largest_total": "Most instructions",
		"smallest_total": "Least instructions",
	}.get(overlay_mode or "none", "No overlay")


def _sorted_opcodes(records: List[Dict[str, Any]], opcodes: Sequence[str], sort_mode: str, include_call: bool) -> List[str]:
	filtered = [op for op in opcodes if include_call or op != "call"]
	if sort_mode == "total_desc":
		return sorted(
			filtered,
			key=lambda op: sum(int(record["scaled_counts"].get(op, 0)) for record in records),
			reverse=True,
		)
	if sort_mode == "variance_desc":
		return sorted(
			filtered,
			key=lambda op: _std([math.log1p(int(record["scaled_counts"].get(op, 0))) for record in records]),
			reverse=True,
		)
	return sorted(filtered)


def _filter_opcodes_by_min_total(
	records: List[Dict[str, Any]],
	opcodes: Sequence[str],
	include_call: bool,
	min_total_value: Any,
) -> List[str]:
	try:
		min_total = max(0, float(min_total_value or 0))
	except (TypeError, ValueError):
		min_total = 0.0
	if min_total <= 0:
		return list(opcodes)
	return [
		op
		for op in opcodes
		if sum(int(_filtered_counts(record, include_call).get(op, 0)) for record in records) >= min_total
	]


def _build_heatmap_matrix(
	records: List[Dict[str, Any]],
	reference_records: List[Dict[str, Any]],
	opcodes: Sequence[str],
	include_call: bool,
	color_mode: str = "mix_plus_total",
) -> List[List[float]]:
	if not records or not opcodes:
		return []
	if color_mode == "portion_only":
		return _build_heatmap_portion_matrix(records, opcodes, include_call)
	return _build_heatmap_mix_plus_total_matrix(records, reference_records, opcodes, include_call)


def _build_heatmap_portion_matrix(
	records: List[Dict[str, Any]],
	opcodes: Sequence[str],
	include_call: bool,
) -> List[List[float]]:
	matrix: List[List[float]] = []
	for record in records:
		counts = _filtered_counts(record, include_call)
		row_total = sum(int(counts.get(op, 0)) for op in opcodes)
		matrix.append(
			[
				0.0 if row_total <= 0 else int(counts.get(op, 0)) / row_total
				for op in opcodes
			]
		)
	return matrix


def _build_heatmap_mix_plus_total_matrix(
	records: List[Dict[str, Any]],
	reference_records: List[Dict[str, Any]],
	opcodes: Sequence[str],
	include_call: bool,
) -> List[List[float]]:
	reference_totals = [math.log1p(_total_for_record(record, include_call)) for record in reference_records]
	min_total = min(reference_totals) if reference_totals else 0.0
	max_total = max(reference_totals) if reference_totals else 0.0
	total_span = max_total - min_total

	matrix: List[List[float]] = []
	for record in records:
		counts = _filtered_counts(record, include_call)
		row_max = max([int(counts.get(op, 0)) for op in opcodes] or [0])
		total_scaled = math.log1p(sum(counts.values()))
		normalized_total = 1.0 if total_span == 0 else (total_scaled - min_total) / total_span
		total_multiplier = 0.25 + 0.75 * max(0.0, min(1.0, normalized_total))
		row = []
		for op in opcodes:
			ratio = 0.0 if row_max <= 0 else int(counts.get(op, 0)) / row_max
			row.append(ratio * total_multiplier)
		matrix.append(row)
	return matrix


def _heatmap_colorbar_title(color_mode: str) -> str:
	return "Opcode share" if color_mode == "portion_only" else "Mix + total"


def _heatmap_colorscale(color_mode: str) -> List[List[Any]]:
	if color_mode == "portion_only":
		return [
			[0.0, "#f8fafc"],
			[0.2, "#fef3c7"],
			[0.45, "#f59e0b"],
			[0.7, "#dc2626"],
			[1.0, "#7f1d1d"],
		]
	return [
		[0.0, "#f8fafc"],
		[0.18, "#d9f99d"],
		[0.42, "#22c55e"],
		[0.68, "#7c3aed"],
		[1.0, "#111827"],
	]


def _empty_figure(message: str) -> go.Figure:
	fig = go.Figure()
	fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
	fig.update_layout(
		height=360,
		margin={"l": 40, "r": 20, "t": 40, "b": 40},
		xaxis={"visible": False},
		yaxis={"visible": False},
	)
	return fig


def _build_heatmap_figure(
	records: List[Dict[str, Any]],
	reference_records: List[Dict[str, Any]],
	opcodes: Sequence[str],
	include_call: bool,
	color_mode: str = "mix_plus_total",
) -> go.Figure:
	if not records:
		return _empty_figure("No instruction-mix rows for this selection.")
	if not opcodes:
		return _empty_figure("No opcodes for this selection.")

	z_values = _build_heatmap_matrix(records, reference_records, opcodes, include_call, color_mode)
	y_labels = [f"{record['exp_id']} {record['param_hash'][:8]}" for record in records]
	customdata = []
	for record in records:
		raw_counts = _filtered_raw_counts(record, include_call)
		scaled_counts = _filtered_counts(record, include_call)
		total_scaled = sum(scaled_counts.values())
		total_raw = sum(raw_counts.values())
		row = []
		for op in opcodes:
			scaled = int(scaled_counts.get(op, 0))
			raw = int(raw_counts.get(op, 0))
			share = 0.0 if total_scaled <= 0 else scaled / total_scaled
			row.append(
				[
					record["exp_id"],
					record["param_hash"],
					_format_runtime(record["runtime_ms"]),
					record["launch_description"],
					raw,
					scaled,
					share,
					total_raw,
					total_scaled,
					record["variant_label"],
				]
			)
		customdata.append(row)

	fig = go.Figure(
		data=go.Heatmap(
			z=z_values,
			x=list(opcodes),
			y=y_labels,
			customdata=customdata,
			colorscale=_heatmap_colorscale(color_mode),
			zmin=0,
			zmax=1,
			colorbar={"title": _heatmap_colorbar_title(color_mode)},
			hovertemplate=(
				"exp_id=%{customdata[0]}<br>"
				"hash=%{customdata[1]}<br>"
				"variant=%{customdata[9]}<br>"
				"runtime=%{customdata[2]} ms<br>"
				"launch=%{customdata[3]}<br>"
				"opcode=%{x}<br>"
				"raw=%{customdata[4]}<br>"
				"scaled=%{customdata[5]}<br>"
				"share=%{customdata[6]:.4f}<br>"
				"raw total=%{customdata[7]}<br>"
				"scaled total=%{customdata[8]}<extra></extra>"
			),
		)
	)
	fig.update_layout(
		height=max(420, min(1200, 110 + len(records) * 11)),
		margin={"l": 120, "r": 20, "t": 30, "b": 90},
		xaxis={"tickangle": -45, "title": "LLVM opcode"},
		yaxis={"title": "Configuration", "autorange": "reversed"},
	)
	return fig


def _histogram_log_edges(values: Sequence[int], bin_count: int) -> List[float]:
	if not values:
		return []
	log_values = [math.log1p(max(0, int(value))) for value in values]
	min_log = min(log_values)
	max_log = max(log_values)
	if min_log == max_log:
		return [min_log, max_log]

	bin_count = max(1, min(80, int(bin_count)))
	width = (max_log - min_log) / bin_count
	return [min_log + index * width for index in range(bin_count + 1)]


def _histogram_bin_rows(values: Sequence[int], bin_count: int, log_edges: Sequence[float] | None = None) -> List[Dict[str, Any]]:
	if not values:
		return []
	edges = list(log_edges or _histogram_log_edges(values, bin_count))
	if len(edges) < 2:
		return []
	if edges[0] == edges[-1]:
		raw_value = int(round(math.expm1(edges[0])))
		return [{"center": edges[0], "frequency": len(values), "low": raw_value, "high": raw_value}]

	bins = [
		{
			"center": (edges[index] + edges[index + 1]) * 0.5,
			"frequency": 0,
			"low": 0,
			"high": 0,
		}
		for index in range(len(edges) - 1)
	]
	for index, row in enumerate(bins):
		row["low"] = int(math.floor(math.expm1(edges[index])))
		row["high"] = int(math.ceil(math.expm1(edges[index + 1])))

	for value in values:
		log_value = math.log1p(max(0, int(value)))
		if log_value <= edges[0]:
			index = 0
		elif log_value >= edges[-1]:
			index = len(bins) - 1
		else:
			index = next(edge_index for edge_index in range(len(edges) - 1) if edges[edge_index] <= log_value < edges[edge_index + 1])
		bins[index]["frequency"] += 1
	return bins


def _build_histogram_figure(
	records: List[Dict[str, Any]],
	opcodes: Sequence[str],
	include_call: bool,
	bin_count_value: Any,
	overlay_records: List[Dict[str, Any]] | None = None,
	overlay_label: str = "Overlay",
) -> go.Figure:
	if not records or not opcodes:
		return _empty_figure("No histogram data for this selection.")
	opcodes = [
		op
		for op in opcodes
		if sum(int(_filtered_counts(record, include_call).get(op, 0)) for record in records) > 0
	]
	if not opcodes:
		return _empty_figure("No nonzero opcode counts for this histogram selection.")
	try:
		bin_count = max(4, min(80, int(bin_count_value or 24)))
	except (TypeError, ValueError):
		bin_count = 24

	cols = 3
	rows = int(math.ceil(len(opcodes) / cols))
	fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(opcodes), horizontal_spacing=0.08, vertical_spacing=0.08)
	for index, op in enumerate(opcodes):
		values = [int(_filtered_counts(record, include_call).get(op, 0)) for record in records]
		overlay_values = [
			int(_filtered_counts(record, include_call).get(op, 0))
			for record in (overlay_records or [])
		]
		bin_edges = _histogram_log_edges(values + overlay_values, bin_count)
		bins = _histogram_bin_rows(values, bin_count, bin_edges)
		overlay_bins = _histogram_bin_rows(overlay_values, bin_count, bin_edges) if overlay_values else []
		row_index = index // cols + 1
		col_index = index % cols + 1
		fig.add_trace(
			go.Bar(
				x=[bin_row["center"] for bin_row in bins],
				y=[bin_row["frequency"] for bin_row in bins],
				customdata=[[bin_row["low"], bin_row["high"]] for bin_row in bins],
				marker_color="rgba(100, 116, 139, 0.55)",
				name="Current selection",
				hovertemplate="current<br>scaled count %{customdata[0]} to %{customdata[1]}<br>configs=%{y}<extra></extra>",
				showlegend=index == 0,
			),
			row=row_index,
			col=col_index,
		)
		if overlay_bins:
			fig.add_trace(
				go.Bar(
					x=[bin_row["center"] for bin_row in overlay_bins],
					y=[bin_row["frequency"] for bin_row in overlay_bins],
					customdata=[[bin_row["low"], bin_row["high"]] for bin_row in overlay_bins],
					marker_color="rgba(220, 38, 38, 0.68)",
					name=overlay_label,
					hovertemplate=f"{overlay_label}<br>scaled count %{{customdata[0]}} to %{{customdata[1]}}<br>configs=%{{y}}<extra></extra>",
					showlegend=index == 0,
				),
				row=row_index,
				col=col_index,
			)
		fig.update_xaxes(title_text="log1p count", row=row_index, col=col_index)
		fig.update_yaxes(title_text="configs", row=row_index, col=col_index)

	fig.update_layout(
		height=max(520, rows * 230),
		margin={"l": 50, "r": 20, "t": 50, "b": 50},
		barmode="overlay",
	)
	return fig


def _build_tuning_param_filter_controls(records: List[Dict[str, Any]], control_type: str) -> List[html.Div]:
	params = _tuning_param_names(records)
	if not params:
		return [html.Div("No tuning-parameter filters for this selection.", className="muted")]
	controls = []
	for param in params:
		values = _tuning_param_values(records, param)
		controls.append(
			html.Div(
				[
					html.Div(param, className="filter-label"),
					dcc.Dropdown(
						id={"type": control_type, "param": param},
						options=[{"label": "All values", "value": ALL_TUNING_PARAM_VALUES}]
						+ [{"label": value, "value": value} for value in values],
						value=ALL_TUNING_PARAM_VALUES,
						clearable=False,
						searchable=False,
					),
				]
			)
		)
	return controls


def _detail_for_exp_id(records: List[Dict[str, Any]], exp_id: int | None, include_call: bool) -> List[html.Div]:
	if not records:
		return [html.Div("No selected configuration.", className="muted")]
	record = None
	if exp_id is not None:
		record = next((candidate for candidate in records if int(candidate["exp_id"]) == int(exp_id)), None)
	record = record or records[0]
	counts = _filtered_counts(record, include_call)
	top_ops = sorted(counts.items(), key=lambda item: int(item[1]), reverse=True)[:8]
	config_preview = json.dumps(json.loads(record["config_json"]), indent=2)[:2400]
	return [
		html.Div(
			className="detail-grid",
			children=[
				html.Div([html.Div("exp_id", className="summary-label"), html.Div(str(record["exp_id"]), className="detail-value")]),
				html.Div([html.Div("Runtime ms", className="summary-label"), html.Div(_format_runtime(record["runtime_ms"]), className="detail-value")]),
				html.Div([html.Div("Variant", className="summary-label"), html.Div(record["variant_label"], className="detail-value")]),
				html.Div([html.Div("Total scaled", className="summary-label"), html.Div(_format_count(sum(counts.values())), className="detail-value")]),
			],
		),
		html.Div(f"Hash: {record['param_hash']}", className="muted"),
		html.Div(f"Launch: {record['launch_description']}", className="muted"),
		html.Div("Top opcodes: " + ", ".join(f"{op}:{_format_count(value)}" for op, value in top_ops), className="muted"),
		html.Pre(config_preview, className="config-preview"),
	]


def _ensure_deep_dive_artifacts(db_path: Path, dataset: Dict[str, Any]) -> Path:
	output_dir = db_path.parent / DEEP_DIVE_RELATIVE_DIR
	output_dir.mkdir(parents=True, exist_ok=True)
	_write_analysis_log(output_dir / "analysis.log", db_path)
	_write_run_manifest(output_dir / "run_manifest.json", db_path, dataset)
	_write_dataset_summary(output_dir / "dataset_summary.csv", dataset)
	_write_opcode_summary(output_dir / "opcode_summary.csv", dataset)
	return output_dir


def _write_analysis_log(path: Path, db_path: Path) -> None:
	lines = [
		f"created_at={datetime.now().isoformat()}",
		f"db_path={db_path}",
		"source_rows=experiments joined with llvm_instruction_counts",
		"row_filter=status='completed' and available instruction-count rows",
		"failed_rows=excluded because they have no trustworthy runtime/instruction-count pair",
		"scaling=scaled_count[op] = raw total_instruction_counts_json[op] * launched thread count",
		"gaussian_launch=local sizes from wi_1/wi_2 dimensions; num groups from wg_1/wg_2 dimensions",
		"gemm_launch=local sizes from num_wi_l_1/num_wi_l_2/num_wi_r_1; num groups from num_wg_*; gemm_2 uses one r_1 workgroup",
		"opcode_policy=call is preserved in the data and controlled by an interactive include/exclude toggle",
		"gemm_policy=primary is gemm_1, combined is gemm_1 plus gemm_2 after per-template launch scaling, gemm_2 remains inspectable",
		"heatmap_color=interactive mode chooses either true per-row opcode share or row_max opcode ratio * (0.25 + 0.75 * normalized_log_total_scaled_count)",
		"histogram_policy=per-op small multiples over log1p scaled-count bins; hover reports raw scaled-count bin ranges",
		"tuning_param_filter_policy=heatmap and histogram each expose single-value filters for every tuning parameter in the selected kernel/input/variant group; defaults keep all values",
		"rationale=instruction mix is stored per work item/program path, so launch scaling keeps configurations comparable by launched work volume",
	]
	path.write_text("\n".join(lines) + "\n")


def _write_run_manifest(path: Path, db_path: Path, dataset: Dict[str, Any]) -> None:
	payload = {
		"created_at": datetime.now().isoformat(),
		"db_path": str(db_path),
		"output_dir": str(path.parent),
		"script": str(Path(__file__).resolve()),
		"completed_variant_records": len(dataset["records"]),
		"kernels": dataset["kernels"],
		"sizes_by_kernel": dataset["sizes_by_kernel"],
		"variants_by_kernel": dataset["variants_by_kernel"],
		"opcodes": dataset["opcodes"],
		"group_counts": dataset["group_counts"],
		"template_counts": dataset["template_counts"],
	}
	path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_dataset_summary(path: Path, dataset: Dict[str, Any]) -> None:
	fields = [
		"kernel_type",
		"input_size",
		"variant",
		"record_count",
		"runtime_min_ms",
		"runtime_median_ms",
		"runtime_max_ms",
		"launch_threads_min",
		"launch_threads_median",
		"launch_threads_max",
		"scaled_total_min",
		"scaled_total_median",
		"scaled_total_max",
	]
	rows = []
	for kernel in dataset["kernels"]:
		for size in dataset["sizes_by_kernel"].get(kernel, []):
			for variant in dataset["variants_by_kernel"].get(kernel, []):
				records = _records_for_group(dataset, kernel, size, variant)
				if not records:
					continue
				runtimes = sorted(float(record["runtime_ms"] or 0.0) for record in records)
				launches = sorted(float(record["launch_threads"]) for record in records)
				totals = sorted(float(sum(record["scaled_counts"].values())) for record in records)
				rows.append(
					{
						"kernel_type": kernel,
						"input_size": size,
						"variant": variant,
						"record_count": len(records),
						"runtime_min_ms": runtimes[0],
						"runtime_median_ms": _percentile(runtimes, 0.5),
						"runtime_max_ms": runtimes[-1],
						"launch_threads_min": launches[0],
						"launch_threads_median": _percentile(launches, 0.5),
						"launch_threads_max": launches[-1],
						"scaled_total_min": totals[0],
						"scaled_total_median": _percentile(totals, 0.5),
						"scaled_total_max": totals[-1],
					}
				)

	with path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)


def _write_opcode_summary(path: Path, dataset: Dict[str, Any]) -> None:
	fields = [
		"kernel_type",
		"input_size",
		"variant",
		"opcode",
		"record_count",
		"mean",
		"median",
		"std",
		"p05",
		"p95",
		"zero_count",
		"cv",
	]
	rows = []
	for kernel in dataset["kernels"]:
		for size in dataset["sizes_by_kernel"].get(kernel, []):
			for variant in dataset["variants_by_kernel"].get(kernel, []):
				records = _records_for_group(dataset, kernel, size, variant)
				if not records:
					continue
				for op in dataset["opcodes"]:
					values = sorted(float(record["scaled_counts"].get(op, 0)) for record in records)
					avg = _mean(values)
					rows.append(
						{
							"kernel_type": kernel,
							"input_size": size,
							"variant": variant,
							"opcode": op,
							"record_count": len(values),
							"mean": avg,
							"median": _percentile(values, 0.5),
							"std": _std(values),
							"p05": _percentile(values, 0.05),
							"p95": _percentile(values, 0.95),
							"zero_count": sum(1 for value in values if value == 0),
							"cv": 0.0 if avg == 0 else _std(values) / avg,
						}
					)

	with path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)


def _build_summary(db_path: Path) -> Dict[str, Any]:
	total = _query_one(db_path, "SELECT COUNT(*) AS v FROM experiments").get("v", 0)
	completed = _query_one(
		db_path,
		"SELECT COUNT(*) AS v FROM experiments WHERE status = ?",
		("completed",),
	).get("v", 0)
	failed = _query_one(
		db_path,
		"SELECT COUNT(*) AS v FROM experiments WHERE status = ?",
		("failed",),
	).get("v", 0)
	pending = _query_one(
		db_path,
		"SELECT COUNT(*) AS v FROM experiments WHERE status = ?",
		("pending",),
	).get("v", 0)
	avg_runtime = _query_one(
		db_path,
		"SELECT AVG(runtime_ms) AS v FROM experiments WHERE status = ?",
		("completed",),
	).get("v")

	by_kernel = _query_rows(
		db_path,
		"""
		SELECT kernel_type, COUNT(*) AS total,
		       AVG(CASE WHEN status = 'completed' THEN runtime_ms END) AS avg_runtime_ms
		FROM experiments
		GROUP BY kernel_type
		ORDER BY kernel_type
		""",
	)

	return {
		"total": total,
		"completed": completed,
		"failed": failed,
		"pending": pending,
		"avg_runtime_ms": avg_runtime,
		"by_kernel": by_kernel,
	}


def _build_options(db_path: Path) -> Dict[str, Any]:
	kernels = [
		row["kernel_type"]
		for row in _query_rows(db_path, "SELECT DISTINCT kernel_type FROM experiments ORDER BY kernel_type")
	]
	sizes = [
		row["input_size"]
		for row in _query_rows(db_path, "SELECT DISTINCT input_size FROM experiments ORDER BY input_size")
	]
	statuses = [row["status"] for row in _query_rows(db_path, "SELECT DISTINCT status FROM experiments ORDER BY status")]
	return {"kernel_types": kernels, "input_sizes": sizes, "statuses": statuses}


def _build_results_payload(
	db_path: Path,
	kernel_type: str | None,
	input_size: str | None,
	status: str | None,
	limit_value: Any,
) -> Dict[str, Any]:
	try:
		limit = max(1, min(2000, int(limit_value or 300)))
	except (TypeError, ValueError):
		limit = 300

	where_clauses = []
	params: List[Any] = []
	if kernel_type:
		where_clauses.append("kernel_type = ?")
		params.append(kernel_type)
	if input_size:
		where_clauses.append("input_size = ?")
		params.append(input_size)
	if status:
		where_clauses.append("status = ?")
		params.append(status)

	where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
	sql = f"""
		SELECT exp_id, kernel_type, input_size, param_hash, runtime_ms, status, timestamp, llvm_ir_path
		FROM experiments
		{where_sql}
		ORDER BY exp_id DESC
		LIMIT ?
	"""
	params.append(limit)
	rows = _query_rows(db_path, sql, tuple(params))
	for row in rows:
		llvm_ir_path = row.get("llvm_ir_path")
		if not llvm_ir_path:
			continue
		try:
			decoded = json.loads(llvm_ir_path)
		except (TypeError, json.JSONDecodeError):
			continue
		if isinstance(decoded, list):
			row["llvm_ir_path"] = ", ".join(str(item) for item in decoded)
	return {"rows": rows, "count": len(rows), "limit": limit}


def _build_inst_counts_payload(
	db_path: Path,
	kernel_type: str | None,
	input_size: str | None,
	limit_value: Any = 100,
) -> Dict[str, Any]:
	try:
		limit = max(1, min(1000, int(limit_value or 100)))
	except (TypeError, ValueError):
		limit = 100

	where_clauses = []
	params: List[Any] = []
	if kernel_type:
		where_clauses.append("e.kernel_type = ?")
		params.append(kernel_type)
	if input_size:
		where_clauses.append("e.input_size = ?")
		params.append(input_size)

	where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
	sql = f"""
		SELECT lic.exp_id, e.kernel_type, e.input_size, lic.template_name, lic.kernel_function, lic.llvm_ir_path,
		       lic.bb_counts_json, lic.total_instruction_counts_json
		FROM llvm_instruction_counts lic
		JOIN experiments e ON e.exp_id = lic.exp_id
		{where_sql}
		ORDER BY lic.exp_id DESC, lic.template_name ASC
		LIMIT ?
	"""
	params.append(limit)
	rows = _query_rows(db_path, sql, tuple(params))

	processed: List[Dict[str, Any]] = []
	for row in rows:
		bb_counts = json.loads(row["bb_counts_json"])
		total_counts = json.loads(row["total_instruction_counts_json"])
		total_dynamic = int(sum(int(v) for v in total_counts.values()))
		top_insts = sorted(total_counts.items(), key=lambda item: int(item[1]), reverse=True)[:5]
		processed.append(
			{
				"exp_id": row["exp_id"],
				"kernel_type": row["kernel_type"],
				"input_size": row["input_size"],
				"template_name": row["template_name"],
				"kernel_function": row["kernel_function"],
				"llvm_ir_path": row["llvm_ir_path"],
				"basic_block_count": len(bb_counts),
				"total_dynamic_instruction_count": total_dynamic,
				"top_instructions": ", ".join(f"{k}:{v}" for k, v in top_insts),
			}
		)

	return {"rows": processed, "count": len(processed), "limit": limit}


def _csv_text(rows: Sequence[Dict[str, Any]]) -> str:
	if not rows:
		return ""
	fieldnames = list(rows[0].keys())
	buffer = io.StringIO()
	writer = csv.DictWriter(buffer, fieldnames=fieldnames)
	writer.writeheader()
	writer.writerows(rows)
	return buffer.getvalue()


def _raw_export_rows(db_path: Path, mix_dataset: Dict[str, Any]) -> Dict[str, Any]:
	experiments = _query_rows(
		db_path,
		"""
		SELECT exp_id, kernel_type, input_size, param_hash, config_json, runtime_ms,
		       llvm_ir_path, timestamp, status, error_msg
		FROM experiments
		ORDER BY exp_id ASC
		""",
	)
	instruction_counts = _query_rows(
		db_path,
		"""
		SELECT id, exp_id, template_name, kernel_function, llvm_ir_path,
		       bb_counts_json, bb_instruction_counts_json, total_instruction_counts_json, created_at
		FROM llvm_instruction_counts
		ORDER BY exp_id ASC, template_name ASC
		""",
	)
	experiment_logs = _query_rows(
		db_path,
		"""
		SELECT log_id, exp_id, stage, level, message, payload_json, created_at
		FROM experiment_logs
		ORDER BY log_id ASC
		""",
	)
	return {
		"db_path": str(db_path),
		"exported_at": datetime.now().isoformat(),
		"experiments": experiments,
		"llvm_instruction_counts": instruction_counts,
		"experiment_logs": experiment_logs,
		"instruction_mix_records": mix_dataset["records"],
		"instruction_mix_template_counts": mix_dataset["template_counts"],
	}


def _raw_export_zip_bytes(db_path: Path, mix_dataset: Dict[str, Any]) -> bytes:
	raw_rows = _raw_export_rows(db_path, mix_dataset)
	buffer = io.BytesIO()
	with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
		archive.writestr("README.txt", "\n".join([
			"ATF raw data export",
			f"database={db_path}",
			f"exported_at={raw_rows['exported_at']}",
			"contents:",
			"  - experiments.json / experiments.csv",
			"  - llvm_instruction_counts.json / llvm_instruction_counts.csv",
			"  - experiment_logs.json / experiment_logs.csv",
			"  - instruction_mix_records.json / instruction_mix_records.csv",
			"  - instruction_mix_template_counts.json / instruction_mix_template_counts.csv",
		]) + "\n")
		for key in (
			"experiments",
			"llvm_instruction_counts",
			"experiment_logs",
			"instruction_mix_records",
			"instruction_mix_template_counts",
		):
			rows = raw_rows[key]
			archive.writestr(f"{key}.json", json.dumps(rows, indent=2, sort_keys=True) + "\n")
			archive.writestr(f"{key}.csv", _csv_text(rows))
		manifest = {
			"db_path": raw_rows["db_path"],
			"exported_at": raw_rows["exported_at"],
			"row_counts": {
				"experiments": len(raw_rows["experiments"]),
				"llvm_instruction_counts": len(raw_rows["llvm_instruction_counts"]),
				"experiment_logs": len(raw_rows["experiment_logs"]),
				"instruction_mix_records": len(raw_rows["instruction_mix_records"]),
				"instruction_mix_template_counts": len(raw_rows["instruction_mix_template_counts"]),
			},
		}
		archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
	return buffer.getvalue()


def _format_runtime(value: Any) -> str:
	if value is None:
		return "-"
	try:
		return f"{float(value):.4f}"
	except (TypeError, ValueError):
		return str(value)


def _summary_cards(summary: Dict[str, Any]) -> List[html.Div]:
	items = [
		("Total", summary["total"]),
		("Completed", summary["completed"]),
		("Failed", summary["failed"]),
		("Pending", summary["pending"]),
		("Avg runtime (ms)", _format_runtime(summary["avg_runtime_ms"])),
	]
	return [
		html.Div(
			[
				html.Div(label, className="summary-label"),
				html.Div(str(value), className="summary-value"),
			],
			className="summary-card",
		)
		for label, value in items
	]


def _table_columns(rows: List[Dict[str, Any]], selected_columns: List[str] | None = None) -> List[Dict[str, str]]:
	if not rows:
		return []
	all_columns = list(rows[0].keys())
	visible_columns = [column for column in (selected_columns or all_columns) if column in all_columns]
	return [{"name": column, "id": column} for column in visible_columns]


def _pick_server_port(host: str, preferred_port: int) -> tuple[int, bool]:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			sock.bind((host, preferred_port))
			return preferred_port, False
		except OSError:
			sock.bind((host, 0))
			return int(sock.getsockname()[1]), True


def _display_urls(host: str, port: int) -> List[str]:
	if host not in {"0.0.0.0", "::"}:
		return [f"http://{host}:{port}"]

	urls = [f"http://127.0.0.1:{port}"]
	try:
		infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
	except OSError:
		infos = []
	for info in infos:
		address = info[4][0]
		if address.startswith("127."):
			continue
		url = f"http://{address}:{port}"
		if url not in urls:
			urls.append(url)
	return urls


def create_dash_app(db_path: Path) -> Dash:
	if DASH_IMPORT_ERROR is not None:
		raise SystemExit(
			"Dash and Plotly are required for this visualization UI. Install dependencies with "
			"`pip install -r requirements.txt`."
		) from DASH_IMPORT_ERROR

	summary = _build_summary(db_path)
	options = _build_options(db_path)
	mix_dataset = _load_instruction_mix_dataset(str(db_path.resolve()))
	mix_output_dir = _ensure_deep_dive_artifacts(db_path, mix_dataset)
	default_mix_kernel, default_mix_size, default_mix_variant = _default_mix_selection(mix_dataset)
	app = Dash(__name__)
	app.title = "ATF Experiment Visualization"

	app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body {
        font-family: "Inter", "Segoe UI", sans-serif;
        margin: 0;
        background: #f8fafc;
        color: #0f172a;
      }
      .page {
        max-width: 1400px;
        margin: 0 auto;
        padding: 24px;
      }
      .subtitle {
        color: #475569;
        margin-bottom: 20px;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 24px;
      }
      .summary-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      .summary-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .summary-value {
        margin-top: 8px;
        font-size: 26px;
        font-weight: 700;
      }
      .panel {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      .filters {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        align-items: end;
        margin-bottom: 14px;
      }
      .filter-label {
        font-size: 13px;
        color: #475569;
        margin-bottom: 6px;
      }
      .controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 12px;
      }
      .muted {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 10px;
      }
      .section-title {
        margin: 0 0 12px 0;
      }
      .columns-panel {
        margin-bottom: 12px;
      }
      .detail-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-bottom: 12px;
      }
      .detail-value {
        font-size: 18px;
        font-weight: 650;
        margin-top: 4px;
      }
      .config-preview {
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        overflow: auto;
        max-height: 360px;
        font-size: 12px;
      }
      .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
      .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
        font-family: "SFMono-Regular", Consolas, monospace;
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <script>
      (function () {
        const shortcutIds = new Set([
          "mix-kernel-filter",
          "mix-size-filter",
          "mix-variant-filter",
          "mix-call-toggle",
          "mix-heatmap-focus-mode",
          "mix-heatmap-row-limit",
          "mix-heatmap-row-sort",
          "mix-heatmap-opcode-sort",
          "mix-heatmap-min-op-total",
          "mix-heatmap-color-mode",
          "mix-hist-kernel-filter",
          "mix-hist-size-filter",
          "mix-hist-variant-filter",
          "mix-hist-call-toggle",
          "mix-hist-focus-mode",
          "mix-hist-overlay-mode",
          "mix-hist-overlay-count",
          "mix-hist-row-limit",
          "mix-hist-opcode-sort",
          "mix-hist-bins"
        ]);
        window.__mixShortcutPayload = {seq: 0};
        window.__mixShortcutActiveId = null;

        function shortcutIdFromElementId(id) {
          if (!id) {
            return null;
          }
          if (shortcutIds.has(id)) {
            return id;
          }
          try {
            const parsed = JSON.parse(id);
            if (
              parsed &&
              (parsed.type === "mix-heatmap-param-filter" || parsed.type === "mix-hist-param-filter") &&
              parsed.param
            ) {
              return id;
            }
          } catch (_error) {
            return null;
          }
          return null;
        }

        function controlIdFrom(target) {
          let element = target;
          while (element && element !== document.body) {
            const shortcutId = shortcutIdFromElementId(element.id);
            if (shortcutId) {
              return shortcutId;
            }
            element = element.parentElement;
          }
          return null;
        }

        document.addEventListener("focusin", function (event) {
          const id = controlIdFrom(event.target);
          if (id) {
            window.__mixShortcutActiveId = id;
          }
        }, true);

        document.addEventListener("pointerdown", function (event) {
          const id = controlIdFrom(event.target);
          if (id) {
            window.__mixShortcutActiveId = id;
          }
        }, true);

        document.addEventListener("keydown", function (event) {
          const numericIds = new Set([
            "mix-heatmap-row-limit",
            "mix-heatmap-min-op-total",
            "mix-hist-overlay-count",
            "mix-hist-row-limit",
            "mix-hist-bins"
          ]);
          if (event.altKey || event.ctrlKey || event.metaKey) {
            return;
          }
          const activeId = window.__mixShortcutActiveId || controlIdFrom(event.target);
          if (!activeId) {
            return;
          }
          const key = event.key;
          const isNumber = /^[1-9]$/.test(key);
          if (key !== "ArrowUp" && key !== "ArrowDown" && !isNumber) {
            return;
          }
          const tagName = (event.target.tagName || "").toLowerCase();
          if (isNumber && tagName === "input" && numericIds.has(activeId)) {
            return;
          }
          event.preventDefault();
          const payload = {
            seq: (window.__mixShortcutPayload.seq || 0) + 1,
            target: activeId,
            key: key
          };
          window.__mixShortcutPayload = payload;
          const input = document.getElementById("mix-shortcut-input");
          if (input) {
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
            setter.call(input, JSON.stringify(payload));
            input.dispatchEvent(new Event("input", {bubbles: true}));
            input.dispatchEvent(new Event("change", {bubbles: true}));
          }
        }, true);
      })();
    </script>
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>"""

	app.layout = html.Div(
		className="page",
		children=[
			html.H2("ATF Experiment Visualization"),
			html.Div(f"Database: {db_path}", className="subtitle"),
			html.Div(f"Instruction mix artifacts: {mix_output_dir}", className="subtitle"),
			dcc.Store(id="results-store"),
			dcc.Store(id="inst-store"),
			dcc.Store(id="mix-shortcut-applied-seq", data=0),
			dcc.Download(id="raw-data-download"),
			dcc.Input(id="mix-shortcut-input", type="text", value="", style={"display": "none"}),
			html.Div(_summary_cards(summary), className="summary-grid"),
			html.Div(
				className="panel",
				children=[
					html.H3("Instruction-Mix Deep Dive", className="section-title"),
					html.Div("Dataset", className="filter-label"),
					html.Div(
						className="filters",
						children=[
							html.Div(
								[
									html.Div("Kernel", className="filter-label"),
									dcc.Dropdown(
										id="mix-kernel-filter",
										options=[{"label": value, "value": value} for value in mix_dataset["kernels"]],
										value=default_mix_kernel,
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Input size", className="filter-label"),
									dcc.Dropdown(
										id="mix-size-filter",
										options=[
											{"label": value, "value": value}
											for value in mix_dataset["sizes_by_kernel"].get(default_mix_kernel, [])
										],
										value=default_mix_size,
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("GEMM variant", className="filter-label"),
									dcc.Dropdown(
										id="mix-variant-filter",
										options=[
											{"label": _variant_label(value), "value": value}
											for value in mix_dataset["variants_by_kernel"].get(default_mix_kernel, [])
										],
										value=default_mix_variant,
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Opcode scope", className="filter-label"),
									dcc.Checklist(
										id="mix-call-toggle",
										options=[{"label": "include call", "value": "call"}],
										value=["call"],
										inline=True,
									),
								]
							),
						],
					),
					html.Div(id="mix-info", className="muted"),
					html.H4("Heatmap", className="section-title"),
					html.Div(
						className="filters",
						children=[
							html.Div(
								[
									html.Div("Heatmap focus", className="filter-label"),
									dcc.Dropdown(
										id="mix-heatmap-focus-mode",
										options=[
											{"label": "All rows", "value": "all"},
											{"label": "Longest-running", "value": "long_runtime"},
											{"label": "Shortest-running", "value": "short_runtime"},
											{"label": "Largest total count", "value": "large_total"},
											{"label": "Smallest total count", "value": "small_total"},
										],
										value="all",
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Heatmap rows", className="filter-label"),
									dcc.Input(id="mix-heatmap-row-limit", type="number", min=1, max=MAX_HEATMAP_ROWS, step=1, value=300),
								]
							),
							html.Div(
								[
									html.Div("Heatmap row sort", className="filter-label"),
									dcc.Dropdown(
										id="mix-heatmap-row-sort",
										options=[
											{"label": "Experiment ID", "value": "exp_id"},
											{"label": "Runtime desc", "value": "runtime_desc"},
											{"label": "Runtime asc", "value": "runtime_asc"},
											{"label": "Total count desc", "value": "scaled_desc"},
											{"label": "Total count asc", "value": "scaled_asc"},
										],
										value="runtime_desc",
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Heatmap opcode sort", className="filter-label"),
									dcc.Dropdown(
										id="mix-heatmap-opcode-sort",
										options=[
											{"label": "Name", "value": "name"},
											{"label": "Total count", "value": "total_desc"},
											{"label": "Variance", "value": "variance_desc"},
										],
										value="total_desc",
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Min opcode total", className="filter-label"),
									dcc.Input(
										id="mix-heatmap-min-op-total",
										type="number",
										min=0,
										step=1,
										value=100,
									),
								]
							),
							html.Div(
								[
									html.Div("Heatmap color", className="filter-label"),
									dcc.Dropdown(
										id="mix-heatmap-color-mode",
										options=[
											{"label": "Portion + total count", "value": "mix_plus_total"},
											{"label": "Portion only", "value": "portion_only"},
										],
										value="portion_only",
										clearable=False,
										searchable=False,
									),
								]
							),
						],
					),
					html.Div("Heatmap tuning-parameter filters", className="filter-label"),
					html.Div(id="mix-heatmap-param-filters", className="filters"),
					dcc.Graph(id="mix-heatmap", config={"displayModeBar": True}),
					html.H4("Per-Opcode Count Histograms", className="section-title"),
					html.Div(
						className="filters",
						children=[
							html.Div(
								[
									html.Div("Histogram kernel", className="filter-label"),
									dcc.Dropdown(
										id="mix-hist-kernel-filter",
										options=[{"label": value, "value": value} for value in mix_dataset["kernels"]],
										value=default_mix_kernel,
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Histogram input size", className="filter-label"),
									dcc.Dropdown(
										id="mix-hist-size-filter",
										options=[
											{"label": value, "value": value}
											for value in mix_dataset["sizes_by_kernel"].get(default_mix_kernel, [])
										],
										value=default_mix_size,
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Histogram GEMM variant", className="filter-label"),
									dcc.Dropdown(
										id="mix-hist-variant-filter",
										options=[
											{"label": _variant_label(value), "value": value}
											for value in mix_dataset["variants_by_kernel"].get(default_mix_kernel, [])
										],
										value=default_mix_variant,
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Histogram opcode scope", className="filter-label"),
									dcc.Checklist(
										id="mix-hist-call-toggle",
										options=[{"label": "include call", "value": "call"}],
										value=["call"],
										inline=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Histogram focus", className="filter-label"),
									dcc.Dropdown(
										id="mix-hist-focus-mode",
										options=[
											{"label": "All rows", "value": "all"},
											{"label": "Longest-running", "value": "long_runtime"},
											{"label": "Shortest-running", "value": "short_runtime"},
											{"label": "Largest total count", "value": "large_total"},
											{"label": "Smallest total count", "value": "small_total"},
										],
										value="all",
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Overlay group", className="filter-label"),
									dcc.Dropdown(
										id="mix-hist-overlay-mode",
										options=[
											{"label": "No overlay", "value": "none"},
											{"label": "Fastest kernels", "value": "fastest"},
											{"label": "Slowest kernels", "value": "slowest"},
											{"label": "Most instructions", "value": "largest_total"},
											{"label": "Least instructions", "value": "smallest_total"},
										],
										value="none",
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Overlay kernels", className="filter-label"),
									dcc.Input(id="mix-hist-overlay-count", type="number", min=1, max=MAX_HISTOGRAM_ROWS, step=1, value=100),
								]
							),
							html.Div(
								[
									html.Div("Histogram rows", className="filter-label"),
									dcc.Input(id="mix-hist-row-limit", type="number", min=1, max=MAX_HISTOGRAM_ROWS, step=1, value=5000),
								]
							),
							html.Div(
								[
									html.Div("Histogram opcode sort", className="filter-label"),
									dcc.Dropdown(
										id="mix-hist-opcode-sort",
										options=[
											{"label": "Name", "value": "name"},
											{"label": "Total count", "value": "total_desc"},
											{"label": "Variance", "value": "variance_desc"},
										],
										value="name",
										clearable=False,
										searchable=False,
									),
								]
							),
							html.Div(
								[
									html.Div("Histogram bins", className="filter-label"),
									dcc.Input(id="mix-hist-bins", type="number", min=4, max=80, step=1, value=80),
								]
							),
						],
					),
					html.Div("Histogram tuning-parameter filters", className="filter-label"),
					html.Div(id="mix-hist-param-filters", className="filters"),
					dcc.Graph(id="mix-histograms", config={"displayModeBar": True}),
				],
			),
			html.Div(
				className="panel",
				children=[
					html.H3("Filters", className="section-title"),
					html.Div(
						className="filters",
						children=[
							html.Div(
								[
									html.Div("Kernel", className="filter-label"),
									dcc.Dropdown(
										id="kernel-filter",
										options=[{"label": value, "value": value} for value in options["kernel_types"]],
										placeholder="All kernels",
										clearable=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Input size", className="filter-label"),
									dcc.Dropdown(
										id="size-filter",
										options=[{"label": value, "value": value} for value in options["input_sizes"]],
										placeholder="All input sizes",
										clearable=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Status", className="filter-label"),
									dcc.Dropdown(
										id="status-filter",
										options=[{"label": value, "value": value} for value in options["statuses"]],
										placeholder="All statuses",
										clearable=True,
									),
								]
							),
							html.Div(
								[
									html.Div("Row limit", className="filter-label"),
									dcc.Input(id="limit-input", type="number", min=1, max=2000, step=1, value=300),
								]
							),
						],
					),
					html.Div(
						className="controls",
						children=[
							html.Button("Reset filters", id="reset-button", n_clicks=0),
							html.Button("Select all columns", id="cols-all", n_clicks=0),
							html.Button("Select no columns", id="cols-none", n_clicks=0),
						],
					),
				],
			),
			html.Div(
				className="panel",
				children=[
					html.H3("Results", className="section-title"),
					html.Div(
						className="controls",
						children=[
							html.Button("Download full raw data", id="download-raw-data", n_clicks=0),
						],
					),
					html.Div(id="rows-info", className="muted"),
					html.Div(
						className="columns-panel",
						children=[
							html.Div("Shown columns", className="filter-label"),
							dcc.Checklist(id="column-selector", inline=True),
						],
					),
					dash_table.DataTable(
						id="results-table",
						page_action="none",
						sort_action="native",
						filter_action="native",
						style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
						style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
						style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
					),
				],
			),
			html.Div(
				className="panel",
				children=[
					html.H3("LLVM Instruction Count Summary", className="section-title"),
					html.Div(id="inst-info", className="muted"),
					dash_table.DataTable(
						id="inst-table",
						page_action="none",
						sort_action="native",
						filter_action="native",
						style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
						style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
						style_header={"backgroundColor": "#f1f5f9", "fontWeight": "600"},
					),
				],
			),
		],
	)

	@callback(
		Output("mix-size-filter", "options"),
		Output("mix-size-filter", "value"),
		Output("mix-variant-filter", "options"),
		Output("mix-variant-filter", "value"),
		Input("mix-kernel-filter", "value"),
		State("mix-size-filter", "value"),
		State("mix-variant-filter", "value"),
	)
	def _sync_mix_controls(
		kernel_type: str | None,
		current_size: str | None,
		current_variant: str | None,
	):
		sizes = mix_dataset["sizes_by_kernel"].get(kernel_type or "", [])
		variants = mix_dataset["variants_by_kernel"].get(kernel_type or "", [])
		size_value = current_size if current_size in sizes else (sizes[0] if sizes else None)
		variant_value = current_variant if current_variant in variants else ("primary" if "primary" in variants else (variants[0] if variants else None))
		return (
			[{"label": value, "value": value} for value in sizes],
			size_value,
			[{"label": _variant_label(value), "value": value} for value in variants],
			variant_value,
		)

	@callback(
		Output("mix-hist-size-filter", "options"),
		Output("mix-hist-size-filter", "value"),
		Output("mix-hist-variant-filter", "options"),
		Output("mix-hist-variant-filter", "value"),
		Input("mix-hist-kernel-filter", "value"),
		State("mix-hist-size-filter", "value"),
		State("mix-hist-variant-filter", "value"),
	)
	def _sync_mix_hist_controls(
		kernel_type: str | None,
		current_size: str | None,
		current_variant: str | None,
	):
		sizes = mix_dataset["sizes_by_kernel"].get(kernel_type or "", [])
		variants = mix_dataset["variants_by_kernel"].get(kernel_type or "", [])
		size_value = current_size if current_size in sizes else (sizes[0] if sizes else None)
		variant_value = current_variant if current_variant in variants else ("primary" if "primary" in variants else (variants[0] if variants else None))
		return (
			[{"label": value, "value": value} for value in sizes],
			size_value,
			[{"label": _variant_label(value), "value": value} for value in variants],
			variant_value,
		)

	@callback(
		Output("mix-heatmap-param-filters", "children"),
		Input("mix-kernel-filter", "value"),
		Input("mix-size-filter", "value"),
		Input("mix-variant-filter", "value"),
	)
	def _render_heatmap_param_filters(
		kernel_type: str | None,
		input_size: str | None,
		variant: str | None,
	):
		records = _records_for_group(mix_dataset, kernel_type, input_size, variant)
		return _build_tuning_param_filter_controls(records, "mix-heatmap-param-filter")

	@callback(
		Output("mix-hist-param-filters", "children"),
		Input("mix-hist-kernel-filter", "value"),
		Input("mix-hist-size-filter", "value"),
		Input("mix-hist-variant-filter", "value"),
	)
	def _render_hist_param_filters(
		kernel_type: str | None,
		input_size: str | None,
		variant: str | None,
	):
		records = _records_for_group(mix_dataset, kernel_type, input_size, variant)
		return _build_tuning_param_filter_controls(records, "mix-hist-param-filter")

	@callback(
		Output("mix-kernel-filter", "value", allow_duplicate=True),
		Output("mix-size-filter", "value", allow_duplicate=True),
		Output("mix-variant-filter", "value", allow_duplicate=True),
		Output("mix-call-toggle", "value", allow_duplicate=True),
		Output("mix-heatmap-focus-mode", "value", allow_duplicate=True),
		Output("mix-heatmap-row-limit", "value", allow_duplicate=True),
		Output("mix-heatmap-row-sort", "value", allow_duplicate=True),
		Output("mix-heatmap-opcode-sort", "value", allow_duplicate=True),
		Output("mix-heatmap-min-op-total", "value", allow_duplicate=True),
		Output("mix-heatmap-color-mode", "value", allow_duplicate=True),
		Output("mix-hist-kernel-filter", "value", allow_duplicate=True),
		Output("mix-hist-size-filter", "value", allow_duplicate=True),
		Output("mix-hist-variant-filter", "value", allow_duplicate=True),
		Output("mix-hist-call-toggle", "value", allow_duplicate=True),
		Output("mix-hist-focus-mode", "value", allow_duplicate=True),
		Output("mix-hist-overlay-mode", "value", allow_duplicate=True),
		Output("mix-hist-overlay-count", "value", allow_duplicate=True),
		Output("mix-hist-row-limit", "value", allow_duplicate=True),
		Output("mix-hist-opcode-sort", "value", allow_duplicate=True),
		Output("mix-hist-bins", "value", allow_duplicate=True),
		Output({"type": "mix-heatmap-param-filter", "param": ALL}, "value", allow_duplicate=True),
		Output({"type": "mix-hist-param-filter", "param": ALL}, "value", allow_duplicate=True),
		Output("mix-shortcut-applied-seq", "data"),
		Input("mix-shortcut-input", "value"),
		State("mix-shortcut-applied-seq", "data"),
		State("mix-kernel-filter", "value"),
		State("mix-size-filter", "value"),
		State("mix-variant-filter", "value"),
		State("mix-call-toggle", "value"),
		State("mix-heatmap-focus-mode", "value"),
		State("mix-heatmap-row-limit", "value"),
		State("mix-heatmap-row-sort", "value"),
		State("mix-heatmap-opcode-sort", "value"),
		State("mix-heatmap-min-op-total", "value"),
		State("mix-heatmap-color-mode", "value"),
		State("mix-hist-kernel-filter", "value"),
		State("mix-hist-size-filter", "value"),
		State("mix-hist-variant-filter", "value"),
		State("mix-hist-call-toggle", "value"),
		State("mix-hist-focus-mode", "value"),
		State("mix-hist-overlay-mode", "value"),
		State("mix-hist-overlay-count", "value"),
		State("mix-hist-row-limit", "value"),
		State("mix-hist-opcode-sort", "value"),
		State("mix-hist-bins", "value"),
		State({"type": "mix-heatmap-param-filter", "param": ALL}, "value"),
		State({"type": "mix-heatmap-param-filter", "param": ALL}, "id"),
		State({"type": "mix-hist-param-filter", "param": ALL}, "value"),
		State({"type": "mix-hist-param-filter", "param": ALL}, "id"),
		prevent_initial_call=True,
	)
	def _apply_mix_shortcut(
		shortcut_value: str | None,
		applied_seq: Any,
		heatmap_kernel: str | None,
		heatmap_size: str | None,
		heatmap_variant: str | None,
		heatmap_call: List[str] | None,
		heatmap_focus: str | None,
		heatmap_rows: Any,
		heatmap_row_sort: str | None,
		heatmap_opcode_sort: str | None,
		heatmap_min_op_total: Any,
		heatmap_color_mode: str | None,
		hist_kernel: str | None,
		hist_size: str | None,
		hist_variant: str | None,
		hist_call: List[str] | None,
		hist_focus: str | None,
		hist_overlay_mode: str | None,
		hist_overlay_count: Any,
		hist_rows: Any,
		hist_opcode_sort: str | None,
		hist_bins: Any,
		heatmap_param_values: List[Any],
		heatmap_param_ids: List[Dict[str, Any]],
		hist_param_values: List[Any],
		hist_param_ids: List[Dict[str, Any]],
	):
		heatmap_param_values = list(heatmap_param_values or [])
		hist_param_values = list(hist_param_values or [])
		heatmap_param_no_update = [no_update] * len(heatmap_param_values)
		hist_param_no_update = [no_update] * len(hist_param_values)

		def unchanged(seq_value: Any = no_update):
			return (no_update,) * 20 + (heatmap_param_no_update, hist_param_no_update, seq_value)

		if not shortcut_value:
			return unchanged()
		try:
			shortcut = json.loads(shortcut_value)
		except (TypeError, json.JSONDecodeError):
			return unchanged()
		seq = _to_int(shortcut.get("seq"), default=0)
		if seq <= _to_int(applied_seq, default=0):
			return unchanged()

		target = shortcut.get("target")
		key = shortcut.get("key")
		direction = -1 if key == "ArrowUp" else 1
		is_number = isinstance(key, str) and key.isdigit()

		values = [
			heatmap_kernel,
			heatmap_size,
			heatmap_variant,
			heatmap_call or [],
			heatmap_focus,
			heatmap_rows,
			heatmap_row_sort,
			heatmap_opcode_sort,
			heatmap_min_op_total,
			heatmap_color_mode,
			hist_kernel,
			hist_size,
			hist_variant,
			hist_call or [],
			hist_focus,
			hist_overlay_mode,
			hist_overlay_count,
			hist_rows,
			hist_opcode_sort,
			hist_bins,
		]

		def pick(options: Sequence[str], current: str | None) -> str | None:
			if is_number:
				return _number_option(options, str(key), current)
			return _cycle_option(options, current, direction)

		def pick_param_value(
			records: List[Dict[str, Any]],
			param_ids: Sequence[Dict[str, Any]] | None,
			param_values: List[Any],
			param: str,
		) -> List[Any] | None:
			if not param_ids:
				return None
			for index, param_id in enumerate(param_ids):
				if param_id.get("param") != param:
					continue
				options = _tuning_param_shortcut_options(records, param)
				current = param_values[index] if index < len(param_values) else ALL_TUNING_PARAM_VALUES
				updated = list(param_values)
				updated[index] = pick(options, current)
				return updated
			return None

		if target == "mix-kernel-filter":
			values[0] = pick(mix_dataset["kernels"], heatmap_kernel)
			sizes = mix_dataset["sizes_by_kernel"].get(values[0] or "", [])
			variants = mix_dataset["variants_by_kernel"].get(values[0] or "", [])
			values[1] = values[1] if values[1] in sizes else (sizes[0] if sizes else None)
			values[2] = values[2] if values[2] in variants else ("primary" if "primary" in variants else (variants[0] if variants else None))
		elif target == "mix-size-filter":
			values[1] = pick(mix_dataset["sizes_by_kernel"].get(heatmap_kernel or "", []), heatmap_size)
		elif target == "mix-variant-filter":
			values[2] = pick(mix_dataset["variants_by_kernel"].get(heatmap_kernel or "", []), heatmap_variant)
		elif target == "mix-call-toggle":
			values[3] = [] if "call" in (heatmap_call or []) else ["call"]
		elif target == "mix-heatmap-focus-mode":
			values[4] = pick(FOCUS_OPTIONS, heatmap_focus)
		elif target == "mix-heatmap-row-limit":
			values[5] = _step_numeric(heatmap_rows, direction, 1, MAX_HEATMAP_ROWS, 50)
		elif target == "mix-heatmap-row-sort":
			values[6] = pick(ROW_SORT_OPTIONS, heatmap_row_sort)
		elif target == "mix-heatmap-opcode-sort":
			values[7] = pick(OPCODE_SORT_OPTIONS, heatmap_opcode_sort)
		elif target == "mix-heatmap-min-op-total":
			step = max(1, int(abs(_to_int(heatmap_min_op_total, default=0)) * 0.25) or 1_000_000)
			values[8] = max(0, _to_int(heatmap_min_op_total, default=0) + direction * step)
		elif target == "mix-heatmap-color-mode":
			values[9] = pick(HEATMAP_COLOR_MODE_OPTIONS, heatmap_color_mode)
		elif target == "mix-hist-kernel-filter":
			values[10] = pick(mix_dataset["kernels"], hist_kernel)
			sizes = mix_dataset["sizes_by_kernel"].get(values[10] or "", [])
			variants = mix_dataset["variants_by_kernel"].get(values[10] or "", [])
			values[11] = values[11] if values[11] in sizes else (sizes[0] if sizes else None)
			values[12] = values[12] if values[12] in variants else ("primary" if "primary" in variants else (variants[0] if variants else None))
		elif target == "mix-hist-size-filter":
			values[11] = pick(mix_dataset["sizes_by_kernel"].get(hist_kernel or "", []), hist_size)
		elif target == "mix-hist-variant-filter":
			values[12] = pick(mix_dataset["variants_by_kernel"].get(hist_kernel or "", []), hist_variant)
		elif target == "mix-hist-call-toggle":
			values[13] = [] if "call" in (hist_call or []) else ["call"]
		elif target == "mix-hist-focus-mode":
			values[14] = pick(FOCUS_OPTIONS, hist_focus)
		elif target == "mix-hist-overlay-mode":
			values[15] = pick(HISTOGRAM_OVERLAY_OPTIONS, hist_overlay_mode)
		elif target == "mix-hist-overlay-count":
			values[16] = _step_numeric(hist_overlay_count, direction, 1, MAX_HISTOGRAM_ROWS, 50)
		elif target == "mix-hist-row-limit":
			values[17] = _step_numeric(hist_rows, direction, 1, MAX_HISTOGRAM_ROWS, 500)
		elif target == "mix-hist-opcode-sort":
			values[18] = pick(OPCODE_SORT_OPTIONS, hist_opcode_sort)
		elif target == "mix-hist-bins":
			values[19] = _step_numeric(hist_bins, direction, 4, 80, 1)
		else:
			param_target = _shortcut_param_target(target)
			if not param_target:
				return unchanged(seq)
			param = str(param_target["param"])
			if param_target["type"] == "mix-heatmap-param-filter":
				updated = pick_param_value(
					_records_for_group(mix_dataset, heatmap_kernel, heatmap_size, heatmap_variant),
					heatmap_param_ids,
					heatmap_param_values,
					param,
				)
				if updated is None:
					return unchanged(seq)
				return (no_update,) * 20 + (updated, hist_param_no_update, seq)
			updated = pick_param_value(
				_records_for_group(mix_dataset, hist_kernel, hist_size, hist_variant),
				hist_param_ids,
				hist_param_values,
				param,
			)
			if updated is None:
				return unchanged(seq)
			return (no_update,) * 20 + (heatmap_param_no_update, updated, seq)

		return (*values, heatmap_param_no_update, hist_param_no_update, seq)

	@callback(
		Output("mix-info", "children"),
		Output("mix-heatmap", "figure"),
		Output("mix-histograms", "figure"),
		Input("mix-kernel-filter", "value"),
		Input("mix-size-filter", "value"),
		Input("mix-variant-filter", "value"),
		Input("mix-call-toggle", "value"),
		Input("mix-heatmap-focus-mode", "value"),
		Input("mix-heatmap-row-limit", "value"),
		Input("mix-heatmap-row-sort", "value"),
		Input("mix-heatmap-opcode-sort", "value"),
		Input("mix-heatmap-min-op-total", "value"),
		Input("mix-heatmap-color-mode", "value"),
		Input({"type": "mix-heatmap-param-filter", "param": ALL}, "value"),
		Input("mix-hist-kernel-filter", "value"),
		Input("mix-hist-size-filter", "value"),
		Input("mix-hist-variant-filter", "value"),
		Input("mix-hist-call-toggle", "value"),
		Input("mix-hist-focus-mode", "value"),
		Input("mix-hist-overlay-mode", "value"),
		Input("mix-hist-overlay-count", "value"),
		Input("mix-hist-row-limit", "value"),
		Input("mix-hist-opcode-sort", "value"),
		Input("mix-hist-bins", "value"),
		Input({"type": "mix-hist-param-filter", "param": ALL}, "value"),
		Input("mix-shortcut-applied-seq", "data"),
		State({"type": "mix-heatmap-param-filter", "param": ALL}, "id"),
		State({"type": "mix-hist-param-filter", "param": ALL}, "id"),
	)
	def _render_instruction_mix(
		kernel_type: str | None,
		input_size: str | None,
		variant: str | None,
		call_toggle: List[str] | None,
		heatmap_focus_mode: str | None,
		heatmap_row_limit: Any,
		heatmap_row_sort: str | None,
		heatmap_opcode_sort: str | None,
		heatmap_min_op_total: Any,
		heatmap_color_mode: str | None,
		heatmap_param_values: List[Any],
		hist_kernel_type: str | None,
		hist_input_size: str | None,
		hist_variant: str | None,
		hist_call_toggle: List[str] | None,
		hist_focus_mode: str | None,
		hist_overlay_mode: str | None,
		hist_overlay_count: Any,
		hist_row_limit: Any,
		hist_opcode_sort: str | None,
		hist_bins: Any,
		hist_param_values: List[Any],
		_shortcut_seq: Any,
		heatmap_param_ids: List[Dict[str, Any]],
		hist_param_ids: List[Dict[str, Any]],
	):
		heatmap_include_call = "call" in (call_toggle or [])
		hist_include_call = "call" in (hist_call_toggle or [])
		group_records = _records_for_group(mix_dataset, kernel_type, input_size, variant)
		hist_group_records = _records_for_group(mix_dataset, hist_kernel_type, hist_input_size, hist_variant)
		records = _apply_tuning_param_filters(group_records, heatmap_param_ids, heatmap_param_values)
		hist_records = _apply_tuning_param_filters(hist_group_records, hist_param_ids, hist_param_values)
		heatmap_rows, heatmap_limit = _focused_records(
			records,
			heatmap_focus_mode or "all",
			heatmap_row_limit,
			heatmap_row_sort or "exp_id",
			heatmap_include_call,
			MAX_HEATMAP_ROWS,
		)
		histogram_rows, hist_limit = _focused_records(
			hist_records,
			hist_focus_mode or "all",
			hist_row_limit,
			"exp_id",
			hist_include_call,
			MAX_HISTOGRAM_ROWS,
		)
		histogram_overlay_rows, overlay_limit = _overlay_records(
			hist_records,
			hist_overlay_mode,
			hist_overlay_count,
			hist_include_call,
		)
		heatmap_opcodes = _sorted_opcodes(heatmap_rows or records, mix_dataset["opcodes"], heatmap_opcode_sort or "name", heatmap_include_call)
		heatmap_opcodes = _filter_opcodes_by_min_total(
			heatmap_rows or records,
			heatmap_opcodes,
			heatmap_include_call,
			heatmap_min_op_total,
		)
		histogram_opcodes = _sorted_opcodes(histogram_rows or hist_records, mix_dataset["opcodes"], hist_opcode_sort or "name", hist_include_call)
		heatmap_fig = _build_heatmap_figure(
			heatmap_rows,
			records,
			heatmap_opcodes,
			heatmap_include_call,
			heatmap_color_mode or "mix_plus_total",
		)
		histogram_fig = _build_histogram_figure(
			histogram_rows,
			histogram_opcodes,
			hist_include_call,
			hist_bins,
			histogram_overlay_rows,
			_overlay_label(hist_overlay_mode),
		)

		group_count = len(group_records)
		hist_count = len(histogram_rows)
		info = (
			f"Heatmap {kernel_type or '-'} {input_size or '-'} {_variant_label(variant or '-')}: "
			f"heatmap shows {len(heatmap_rows)} row(s) with limit={heatmap_limit}; "
			f"heatmap uses {len(heatmap_opcodes)} opcode column(s); "
			f"color={heatmap_color_mode or 'mix_plus_total'}; "
			f"tuning filters keep {len(records)}/{group_count} row(s), call={'included' if heatmap_include_call else 'excluded'}. "
			f"Histograms {hist_kernel_type or '-'} {hist_input_size or '-'} {_variant_label(hist_variant or '-')}: "
			f"use {hist_count} row(s) with limit={hist_limit}; "
			f"overlay={_overlay_label(hist_overlay_mode)} rows={len(histogram_overlay_rows)} limit={overlay_limit}; "
			f"tuning filters keep {len(hist_records)}/{len(hist_group_records)} row(s), call={'included' if hist_include_call else 'excluded'}."
		)
		return info, heatmap_fig, histogram_fig

	@callback(
		Output("raw-data-download", "data"),
		Input("download-raw-data", "n_clicks"),
		prevent_initial_call=True,
	)
	def _download_raw_data(_n_clicks: int):
		filename = f"{db_path.stem}_raw_export.zip"
		return dcc.send_bytes(lambda stream: stream.write(_raw_export_zip_bytes(db_path, mix_dataset)), filename)

	@callback(
		Output("kernel-filter", "value"),
		Output("size-filter", "value"),
		Output("status-filter", "value"),
		Output("limit-input", "value"),
		Input("reset-button", "n_clicks"),
		prevent_initial_call=True,
	)
	def _reset_filters(_n_clicks: int):
		return None, None, None, 300

	@callback(
		Output("results-store", "data"),
		Input("kernel-filter", "value"),
		Input("size-filter", "value"),
		Input("status-filter", "value"),
		Input("limit-input", "value"),
	)
	def _load_results(
		kernel_type: str | None,
		input_size: str | None,
		status: str | None,
		limit_value: Any,
	):
		return _build_results_payload(db_path, kernel_type, input_size, status, limit_value)

	@callback(
		Output("inst-store", "data"),
		Input("kernel-filter", "value"),
		Input("size-filter", "value"),
	)
	def _load_inst_counts(kernel_type: str | None, input_size: str | None):
		return _build_inst_counts_payload(db_path, kernel_type, input_size)

	@callback(
		Output("column-selector", "options"),
		Output("column-selector", "value"),
		Input("results-store", "data"),
		Input("cols-all", "n_clicks"),
		Input("cols-none", "n_clicks"),
		State("column-selector", "value"),
	)
	def _sync_selected_columns(
		results_payload: Dict[str, Any] | None,
		_cols_all: int,
		_cols_none: int,
		current_columns: List[str] | None,
	):
		rows = (results_payload or {}).get("rows", [])
		all_columns = list(rows[0].keys()) if rows else []
		options = [{"label": column, "value": column} for column in all_columns]
		triggered = ctx.triggered_id

		if not all_columns:
			return options, []
		if triggered == "cols-none":
			return options, []
		if triggered == "cols-all":
			return options, all_columns

		current_columns = current_columns or []
		filtered_columns = [column for column in current_columns if column in all_columns]
		return options, filtered_columns or all_columns

	@callback(
		Output("rows-info", "children"),
		Output("results-table", "columns"),
		Output("results-table", "data"),
		Output("inst-info", "children"),
		Output("inst-table", "columns"),
		Output("inst-table", "data"),
		Input("results-store", "data"),
		Input("inst-store", "data"),
		Input("column-selector", "value"),
	)
	def _render_outputs(
		results_payload: Dict[str, Any] | None,
		inst_payload: Dict[str, Any] | None,
		selected_columns: List[str] | None,
	):
		results_payload = results_payload or {"rows": [], "count": 0, "limit": 300}
		inst_payload = inst_payload or {"rows": [], "count": 0, "limit": 100}
		rows = results_payload["rows"]
		inst_rows = inst_payload["rows"]

		if rows and selected_columns == [] and ctx.triggered_id == "column-selector":
			result_columns = []
		else:
			result_columns = _table_columns(rows, selected_columns)
		inst_columns = _table_columns(inst_rows)

		return (
			f"Showing {results_payload['count']} rows (limit={results_payload['limit']})",
			result_columns,
			rows,
			f"Showing {inst_payload['count']} LLVM instruction-count row(s)",
			inst_columns,
			inst_rows,
		)

	return app


def main() -> int:
	parser = argparse.ArgumentParser(description="Start web visualization server for experiment SQLite database")
	parser.add_argument("--db", default="experiments.db", help="Path to SQLite experiment database")
	parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0; use 127.0.0.1 for local-only)")
	parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
	args = parser.parse_args()

	db_path = Path(args.db).resolve()
	if not db_path.exists():
		raise SystemExit(f"Database not found: {db_path}")

	app = create_dash_app(db_path)
	actual_port, used_fallback = _pick_server_port(args.host, args.port)
	if used_fallback:
		print(f"Port {args.port} is already in use. Falling back to available port {actual_port}.")
	print(f"Visualization server listening on {args.host}:{actual_port}")
	for url in _display_urls(args.host, actual_port):
		print(f"Open: {url}")
	print(f"Using database: {db_path}")
	print("Press Ctrl+C to stop.")
	app.run(host=args.host, port=actual_port, debug=False)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
