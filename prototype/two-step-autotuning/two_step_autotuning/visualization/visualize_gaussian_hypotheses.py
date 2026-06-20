from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_ROOT = Path(
	"prototype/two-step-autotuning/runs/"
	"live_db30000cfg_valid300_seed5_warm3_median5_test40000_mapop24_fulldb_nolifetime_mut64exhaust_tuners2"
)
GAUSSIAN_CASES = ("gaussian_512", "gaussian_1024", "gaussian_2048")
METHODS = ("parameter_tuning", "instruction_map_tuning")
METHOD_LABELS = {
	"parameter_tuning": "Parameter tuning",
	"instruction_map_tuning": "Instruction-map tuning",
}
METHOD_COLORS = {
	"parameter_tuning": "#4c78a8",
	"instruction_map_tuning": "#e45756",
}


def _read_trace(path: Path) -> list[dict[str, str]]:
	with path.open(newline="") as csv_file:
		return list(csv.DictReader(csv_file))


def _runtime(row: dict[str, str]) -> float | None:
	try:
		value = float(row.get("runtime_ms") or "")
	except ValueError:
		return None
	if not np.isfinite(value) or value >= 1.0e8:
		return None
	return value


def _is_valid(row: dict[str, str]) -> bool:
	return _runtime(row) is not None and bool(row.get("param_hash"))


def _unique_sequence(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
	seen = set()
	sequence = []
	best = float("inf")
	for row in rows:
		if not _is_valid(row):
			continue
		param_hash = row["param_hash"]
		if param_hash in seen:
			continue
		seen.add(param_hash)
		runtime = float(_runtime(row))
		best = min(best, runtime)
		sequence.append(
			{
				"runtime_ms": runtime,
				"best_runtime_ms": best,
				"status": row.get("candidate_status", ""),
				"resolver_distance": _float_or_none(row.get("resolver_distance", "")),
			}
		)
	return sequence


def _float_or_none(value: str) -> float | None:
	try:
		result = float(value)
	except ValueError:
		return None
	if not np.isfinite(result):
		return None
	return result


def _trace_stats(run_root: Path, case: str, seed: int, method: str) -> dict[str, Any]:
	method_dir = run_root / case / f"seed_{seed}" / method
	rows = _read_trace(method_dir / "trace.csv")
	summary = json.loads((method_dir / "summary.json").read_text())
	sequence = _unique_sequence(rows)
	status_counts = defaultdict(int)
	for row in rows:
		status_counts[row.get("candidate_status", "")] += 1
	invalid_count = sum(
		count
		for status, count in status_counts.items()
		if "invalid" in status or status in {"compile_failed", "runtime_failed"}
	)
	inner_tuned_unique = sum(1 for item in sequence if item["status"] == "inner_tuned_valid")
	return {
		"case": case,
		"seed": seed,
		"method": method,
		"rows": rows,
		"summary": summary,
		"sequence": sequence,
		"invalid_fraction": invalid_count / len(rows) if rows else 0.0,
		"unique_config_count": int(summary.get("unique_config_count", len(sequence))),
		"valid_evaluation_count": int(summary.get("valid_evaluation_count", 0)),
		"iterations": int(summary.get("iterations", len(rows))),
		"best_runtime_ms": float(summary.get("best_runtime_ms", min((item["runtime_ms"] for item in sequence), default=0.0))),
		"inner_tuned_unique_fraction": inner_tuned_unique / len(sequence) if sequence else 0.0,
	}


def _median_curve(stats: list[dict[str, Any]], ks: list[int]) -> list[float]:
	values_by_k = []
	for k in ks:
		values = []
		for item in stats:
			seq = item["sequence"]
			if not seq:
				continue
			idx = min(k, len(seq)) - 1
			values.append(float(seq[idx]["best_runtime_ms"]))
		values_by_k.append(float(median(values)) if values else np.nan)
	return values_by_k


def _first_unique_runtimes(stats: list[dict[str, Any]], limit: int) -> list[float]:
	values = []
	for item in stats:
		values.extend(float(row["runtime_ms"]) for row in item["sequence"][:limit])
	return values


def _median(values: list[float]) -> float:
	return float(median(values)) if values else 0.0


def _write_summary(path: Path, all_stats: list[dict[str, Any]], ks: list[int]) -> None:
	rows = []
	grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for item in all_stats:
		grouped[(item["case"], item["method"])].append(item)
	for (case, method), stats in sorted(grouped.items()):
		curve = _median_curve(stats, ks)
		rows.append(
			{
				"case": case,
				"method": method,
				"best_runtime_median_ms": _median([item["best_runtime_ms"] for item in stats]),
				"first_10_unique_best_median_ms": curve[ks.index(10)] if 10 in ks else "",
				"first_20_unique_best_median_ms": curve[ks.index(20)] if 20 in ks else "",
				"first_50_unique_best_median_ms": curve[ks.index(50)] if 50 in ks else "",
				"first_50_unique_runtime_median_ms": _median(_first_unique_runtimes(stats, 50)),
				"unique_config_count_median": _median([item["unique_config_count"] for item in stats]),
				"valid_evaluation_count_median": _median([item["valid_evaluation_count"] for item in stats]),
				"invalid_fraction_median": _median([item["invalid_fraction"] for item in stats]),
				"inner_tuned_unique_fraction_median": _median([item["inner_tuned_unique_fraction"] for item in stats]),
			}
		)
	with path.open("w", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def main() -> int:
	parser = argparse.ArgumentParser(description="Visualize hypotheses for fast Gaussian instruction-map convergence.")
	parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
	parser.add_argument("--output-dir", type=Path, default=None)
	args = parser.parse_args()

	output_dir = args.output_dir or args.run_root / "slide_landscape_nn"
	output_dir.mkdir(parents=True, exist_ok=True)

	all_stats = []
	for case in GAUSSIAN_CASES:
		for seed in range(1, 6):
			for method in METHODS:
				all_stats.append(_trace_stats(args.run_root, case, seed, method))

	ks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300]
	_write_summary(output_dir / "gaussian_hypothesis_summary.csv", all_stats, ks)

	fig = plt.figure(figsize=(14.5, 9.2), constrained_layout=True)
	grid = fig.add_gridspec(2, 2)
	ax_curve = fig.add_subplot(grid[0, 0])
	ax_refinement = fig.add_subplot(grid[0, 1])
	ax_invalid = fig.add_subplot(grid[1, 0])
	ax_compress = fig.add_subplot(grid[1, 1])
	x = np.arange(len(GAUSSIAN_CASES))
	width = 0.34

	for case in GAUSSIAN_CASES:
		case_stats = [item for item in all_stats if item["case"] == case]
		for method in METHODS:
			stats = [item for item in case_stats if item["method"] == method]
			label = f"{case.replace('gaussian_', 'G')} {METHOD_LABELS[method]}"
			ax_curve.plot(
				ks,
				_median_curve(stats, ks),
				marker="o",
				linewidth=2,
				color=METHOD_COLORS[method],
				alpha=0.55 if case != "gaussian_2048" else 1.0,
				label=label,
			)
	ax_curve.set_xscale("log")
	ax_curve.set_xlabel("Unique configs evaluated")
	ax_curve.set_ylabel("Median best runtime (ms)")
	ax_curve.set_title("Supported: instruction maps can find good regions early")
	ax_curve.grid(True, alpha=0.25)
	ax_curve.legend(fontsize=8, ncols=2)

	statuses = ["valid", "inner_tuned_valid", "shared_start"]
	status_colors = {"valid": "#59a14f", "inner_tuned_valid": "#f28e2b", "shared_start": "#bab0ac"}
	bottom = np.zeros(len(GAUSSIAN_CASES), dtype=float)
	for status in statuses:
		values = []
		for case in GAUSSIAN_CASES:
			stats = [item for item in all_stats if item["case"] == case and item["method"] == "instruction_map_tuning"]
			count = sum(1 for item in stats if item["summary"].get("best_row", {}).get("candidate_status") == status)
			values.append(count)
		ax_refinement.bar(
			x,
			values,
			bottom=bottom,
			label=status,
			color=status_colors[status],
			alpha=0.86,
		)
		bottom += np.array(values, dtype=float)
	ax_refinement.set_xticks(x, [case.replace("gaussian_", "G") for case in GAUSSIAN_CASES])
	ax_refinement.set_ylabel("Seeds where best config came from status")
	ax_refinement.set_title("Supported: inner parameter refinement often supplies the winner")
	ax_refinement.set_ylim(0, 5)
	ax_refinement.grid(True, axis="y", alpha=0.25)
	ax_refinement.legend()

	for offset, method in [(-width / 2, "parameter_tuning"), (width / 2, "instruction_map_tuning")]:
		values = [
			_median([item["invalid_fraction"] for item in all_stats if item["case"] == case and item["method"] == method])
			for case in GAUSSIAN_CASES
		]
		ax_invalid.bar(x + offset, values, width, label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.78)
	ax_invalid.set_xticks(x, [case.replace("gaussian_", "G") for case in GAUSSIAN_CASES])
	ax_invalid.set_ylabel("Invalid/precheck-failed fraction")
	ax_invalid.set_title("Supported: parameter search spends effort outside valid space")
	ax_invalid.grid(True, axis="y", alpha=0.25)
	ax_invalid.legend()

	for offset, method in [(-width / 2, "parameter_tuning"), (width / 2, "instruction_map_tuning")]:
		reuse_values = []
		for case in GAUSSIAN_CASES:
			stats = [item for item in all_stats if item["case"] == case and item["method"] == method]
			reuse_values.append(
				_median(
					[
						item["valid_evaluation_count"] / max(item["unique_config_count"], 1)
						for item in stats
					]
				)
			)
		ax_compress.bar(
			x + offset,
			reuse_values,
			width,
			label=METHOD_LABELS[method],
			color=METHOD_COLORS[method],
			alpha=0.78,
		)
	ax_compress.set_xticks(x, [case.replace("gaussian_", "G") for case in GAUSSIAN_CASES])
	ax_compress.set_ylabel("Valid evaluations per unique config")
	ax_compress.set_title("Supported: many instruction requests resolve to a compact config set")
	ax_compress.grid(True, axis="y", alpha=0.25)
	ax_compress.legend()

	fig.suptitle("Why Gaussian instruction-map tuning can converge quickly", fontsize=16)
	output_path = output_dir / "gaussian_hypothesis_diagnostics.png"
	fig.savefig(output_path, dpi=220)
	plt.close(fig)
	print(f"Wrote {output_path}")
	print(f"Wrote {output_dir / 'gaussian_hypothesis_summary.csv'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
