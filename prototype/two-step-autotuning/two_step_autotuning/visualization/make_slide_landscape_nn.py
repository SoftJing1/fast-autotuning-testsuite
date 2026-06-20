from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..core.dataset import TuningDataset
from ..core.instruction_space import filter_instruction_opcodes
from ..core.parameter_space import build_parameter_index_specs, indices_from_config


DEFAULT_RUN_ROOT = Path(
	"prototype/two-step-autotuning/runs/"
	"live_db30000cfg_valid300_seed5_warm3_median5_test40000_mapop24_fulldb_nolifetime_mut64exhaust_tuners2"
)


def _rankdata(values: np.ndarray) -> np.ndarray:
	order = np.argsort(values, kind="mergesort")
	ranks = np.empty(len(values), dtype=float)
	i = 0
	while i < len(values):
		j = i + 1
		while j < len(values) and values[order[j]] == values[order[i]]:
			j += 1
		ranks[order[i:j]] = (i + j - 1) / 2.0
		i = j
	return ranks


def _spearman_abs(x: np.ndarray, y: np.ndarray) -> float:
	if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
		return 0.0
	rx = _rankdata(x)
	ry = _rankdata(y)
	corr = np.corrcoef(rx, ry)[0, 1]
	if not np.isfinite(corr):
		return 0.0
	return abs(float(corr))


def _safe_name(value: str) -> str:
	return value.replace("x", "_").replace("/", "_").replace(".", "_")


def _percentile(values: np.ndarray, pct: float) -> float:
	if len(values) == 0:
		return 0.0
	return float(np.percentile(values, pct))


def _case_name(kernel: str, input_size: str) -> str:
	if kernel == "gaussian":
		return f"gaussian_{input_size.split('x')[0]}"
	if kernel == "gemm":
		return f"gemm_{input_size.split('x')[0]}"
	return f"{kernel}_{_safe_name(input_size)}"


def _load_manifest_cases(run_root: Path) -> tuple[Path, list[tuple[str, str]]]:
	manifest = json.loads((run_root / "experiment_manifest.json").read_text())
	db_path = Path(manifest["database"]["resolver_db"])
	cases = [(item["kernel"], item["input_size"]) for item in manifest["cases"]]
	return db_path, cases


def _sample_indices(count: int, sample_count: int, seed: int) -> np.ndarray:
	rng = random.Random(seed)
	indices = list(range(count))
	if sample_count < count:
		indices = rng.sample(indices, sample_count)
	return np.array(sorted(indices), dtype=int)


def _parameter_matrix(dataset: TuningDataset) -> tuple[np.ndarray, list[str]]:
	specs = build_parameter_index_specs(dataset)
	names = sorted(specs)
	rows = []
	for record in dataset.records:
		encoded = indices_from_config(record.config, specs)
		rows.append([encoded[f"idx.{name}"] for name in names])
	return np.array(rows, dtype=float), names


def _instruction_matrix(dataset: TuningDataset) -> tuple[np.ndarray, list[str]]:
	opcodes = filter_instruction_opcodes(dataset.opcodes)
	rows = []
	for record in dataset.records:
		rows.append([float(record.raw_counts.get(op, 0)) for op in opcodes])
	return np.array(rows, dtype=float), opcodes


def _choose_parameter_axes(matrix: np.ndarray, names: list[str]) -> tuple[int, int]:
	scores = []
	for idx, name in enumerate(names):
		unique_count = len(np.unique(matrix[:, idx]))
		if unique_count < 2:
			continue
		scores.append((unique_count, float(np.var(matrix[:, idx])), name, idx))
	scores.sort(reverse=True)
	if len(scores) < 2:
		raise ValueError("Need at least two varying tuning parameters")
	return scores[0][3], scores[1][3]


def _choose_instruction_axes(matrix: np.ndarray, opcodes: list[str], runtimes: np.ndarray) -> tuple[int, int]:
	candidates = []
	for idx, op in enumerate(opcodes):
		values = matrix[:, idx]
		unique_count = len(np.unique(values))
		if unique_count < 5:
			continue
		candidates.append(
			{
				"idx": idx,
				"op": op,
				"runtime_corr": _spearman_abs(values, runtimes),
				"log_var": math.log1p(float(np.var(np.log1p(values)))),
				"unique_count": unique_count,
			}
		)
	if len(candidates) < 2:
		raise ValueError("Need at least two varying instruction counts")

	overall_runtime_scale = max(_percentile(np.abs(runtimes - np.median(runtimes)), 75), 1.0e-9)
	pair_scores = []
	for left_pos, left in enumerate(candidates):
		for right in candidates[left_pos + 1 :]:
			x = matrix[:, left["idx"]]
			y = matrix[:, right["idx"]]
			axis_corr = _spearman_abs(x, y)
			if axis_corr >= 0.92:
				continue
			coords = np.column_stack([np.log1p(x), np.log1p(y)])
			coords_std = np.std(coords, axis=0)
			if np.any(coords_std <= 0.0):
				continue
			coords = (coords - np.mean(coords, axis=0)) / coords_std
			local_diffs = _nearest_neighbor_abs_diffs(coords, runtimes, np.arange(len(runtimes)))
			local_similarity = max(0.0, 1.0 - float(median(local_diffs)) / overall_runtime_scale)
			runtime_signal = max(left["runtime_corr"], right["runtime_corr"]) + 0.5 * min(
				left["runtime_corr"], right["runtime_corr"]
			)
			coverage = math.sqrt(left["log_var"] * right["log_var"])
			score = runtime_signal + 0.35 * local_similarity + 0.05 * coverage - 0.35 * axis_corr
			pair_scores.append(
				(
					score,
					local_similarity,
					runtime_signal,
					-axis_corr,
					left["op"],
					right["op"],
					left["idx"],
					right["idx"],
				)
			)

	if not pair_scores:
		pair_scores = []
		for left_pos, left in enumerate(candidates):
			for right in candidates[left_pos + 1 :]:
				axis_corr = _spearman_abs(matrix[:, left["idx"]], matrix[:, right["idx"]])
				score = left["runtime_corr"] + right["runtime_corr"] - axis_corr
				pair_scores.append((score, 0.0, score, -axis_corr, left["op"], right["op"], left["idx"], right["idx"]))

	pair_scores.sort(reverse=True)
	return pair_scores[0][6], pair_scores[0][7]


def _nearest_neighbor_abs_diffs(matrix: np.ndarray, runtimes: np.ndarray, base_indices: np.ndarray) -> np.ndarray:
	diffs = []
	for base_idx in base_indices:
		delta = matrix - matrix[base_idx]
		distances = np.sqrt(np.sum(delta * delta, axis=1))
		distances[base_idx] = np.inf
		neighbor_idx = int(np.argmin(distances))
		diffs.append(abs(float(runtimes[neighbor_idx]) - float(runtimes[base_idx])))
	return np.array(diffs, dtype=float)


def _summarize_diffs(diffs: np.ndarray) -> dict[str, float]:
	return {
		"median_ms": float(median(diffs)),
		"mean_ms": float(np.mean(diffs)),
		"p75_ms": _percentile(diffs, 75),
		"p90_ms": _percentile(diffs, 90),
	}


def _plot_case(
	output_dir: Path,
	case: tuple[str, str],
	dataset: TuningDataset,
	scatter_indices: np.ndarray,
	nn_indices: np.ndarray,
) -> dict[str, Any]:
	kernel, input_size = case
	runtimes = np.array([float(record.runtime_ms) for record in dataset.records], dtype=float)
	normalized = runtimes / float(np.min(runtimes))

	param_matrix, param_names = _parameter_matrix(dataset)
	inst_matrix_raw, opcodes = _instruction_matrix(dataset)
	inst_matrix = np.log1p(inst_matrix_raw)

	px, py = _choose_parameter_axes(param_matrix[scatter_indices], param_names)
	ix, iy = _choose_instruction_axes(inst_matrix_raw[scatter_indices], opcodes, runtimes[scatter_indices])

	param_nn_diffs = _nearest_neighbor_abs_diffs(param_matrix[nn_indices], runtimes[nn_indices], np.arange(len(nn_indices)))
	inst_nn_diffs = _nearest_neighbor_abs_diffs(inst_matrix[nn_indices], runtimes[nn_indices], np.arange(len(nn_indices)))

	fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
	grid = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.25])
	axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
	box_ax = fig.add_subplot(grid[1, :])

	color_values = normalized[scatter_indices]
	vmin = float(np.percentile(color_values, 1))
	vmax = float(np.percentile(color_values, 99))
	scatter_kwargs = {
		"c": color_values,
		"cmap": "viridis_r",
		"s": 18,
		"alpha": 0.78,
		"linewidths": 0,
		"vmin": vmin,
		"vmax": vmax,
	}

	axes[0].scatter(param_matrix[scatter_indices, px], param_matrix[scatter_indices, py], **scatter_kwargs)
	axes[0].set_title("Tuning-parameter coordinates")
	axes[0].set_xlabel(f"{param_names[px]} index")
	axes[0].set_ylabel(f"{param_names[py]} index")
	axes[0].grid(True, alpha=0.18)

	plot = axes[1].scatter(inst_matrix_raw[scatter_indices, ix], inst_matrix_raw[scatter_indices, iy], **scatter_kwargs)
	axes[1].set_title("Instruction-count coordinates")
	axes[1].set_xlabel(opcodes[ix])
	axes[1].set_ylabel(opcodes[iy])
	axes[1].set_xscale("symlog")
	axes[1].set_yscale("symlog")
	axes[1].grid(True, alpha=0.18)
	colorbar = fig.colorbar(plot, ax=axes, location="right", shrink=0.88)
	colorbar.set_label("Runtime / best runtime, lower is better")

	box_ax.boxplot(
		[param_nn_diffs, inst_nn_diffs],
		tick_labels=["Nearest in parameter space", "Nearest in instruction-map space"],
		showfliers=False,
		patch_artist=True,
		boxprops={"facecolor": "#d9e8f5", "edgecolor": "#315a7d"},
		medianprops={"color": "#9b2c2c", "linewidth": 2},
		whiskerprops={"color": "#315a7d"},
		capprops={"color": "#315a7d"},
	)
	box_ax.set_title("Runtime difference to nearest neighbor")
	box_ax.set_ylabel("Absolute runtime diff (ms)")
	box_ax.grid(True, axis="y", alpha=0.25)

	fig.suptitle(f"{kernel} {input_size}: parameter space vs instruction-map space", fontsize=15)
	output_path = output_dir / f"{_case_name(kernel, input_size)}_landscape_nn.png"
	fig.savefig(output_path, dpi=220)
	plt.close(fig)

	param_summary = _summarize_diffs(param_nn_diffs)
	inst_summary = _summarize_diffs(inst_nn_diffs)
	return {
		"kernel": kernel,
		"input_size": input_size,
		"records": len(dataset.records),
		"scatter_sample": len(scatter_indices),
		"nn_sample": len(nn_indices),
		"parameter_x": param_names[px],
		"parameter_y": param_names[py],
		"instruction_x": opcodes[ix],
		"instruction_y": opcodes[iy],
		"parameter_nn_median_ms": param_summary["median_ms"],
		"instruction_nn_median_ms": inst_summary["median_ms"],
		"parameter_nn_mean_ms": param_summary["mean_ms"],
		"instruction_nn_mean_ms": inst_summary["mean_ms"],
		"parameter_nn_p90_ms": param_summary["p90_ms"],
		"instruction_nn_p90_ms": inst_summary["p90_ms"],
		"median_ratio_parameter_over_instruction": (
			param_summary["median_ms"] / inst_summary["median_ms"] if inst_summary["median_ms"] > 0 else float("inf")
		),
		"figure": str(output_path),
	}


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	with path.open("w", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Create slide-oriented landscape and nearest-neighbor figures for the two-step autotuning run."
	)
	parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
	parser.add_argument("--case", action="append", default=[])
	parser.add_argument("--scatter-sample", type=int, default=2500)
	parser.add_argument("--nn-sample", type=int, default=1800)
	parser.add_argument("--seed", type=int, default=7)
	args = parser.parse_args()

	db_path, manifest_cases = _load_manifest_cases(args.run_root)
	requested = set(args.case)
	cases = [case for case in manifest_cases if not requested or f"{case[0]}:{case[1]}" in requested]
	output_dir = args.run_root / "slide_landscape_nn"
	output_dir.mkdir(parents=True, exist_ok=True)

	rows = []
	for kernel, input_size in cases:
		dataset = TuningDataset(db_path, kernel, input_size)
		scatter_indices = _sample_indices(len(dataset.records), args.scatter_sample, args.seed)
		nn_indices = _sample_indices(len(dataset.records), args.nn_sample, args.seed + 100)
		row = _plot_case(output_dir, (kernel, input_size), dataset, scatter_indices, nn_indices)
		rows.append(row)
		print(
			f"{kernel} {input_size}: parameter NN median {row['parameter_nn_median_ms']:.4f} ms; "
			f"instruction NN median {row['instruction_nn_median_ms']:.4f} ms; "
			f"ratio {row['median_ratio_parameter_over_instruction']:.2f}; figure {row['figure']}"
		)

	_write_summary(output_dir / "summary.csv", rows)
	(output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
