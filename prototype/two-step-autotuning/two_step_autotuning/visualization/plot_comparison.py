from __future__ import annotations

import argparse
import json
import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from ..core.instruction_space import filter_instruction_opcodes


def _load_series(trace_path: Path, x_axis: str) -> tuple[list[int], list[float]]:
	iterations: list[int] = []
	best_values: list[float] = []
	with trace_path.open() as f:
		for row in csv.DictReader(f):
			value = row.get("best_runtime_ms", "")
			if not value:
				continue
			x_value = row["iteration"]
			if x_axis == "unique_valid":
				x_value = row.get("unique_config_count", "")
				if not x_value:
					continue
			elif x_axis == "valid":
				x_value = row.get("valid_evaluation_count", "") or row.get("unique_config_count", "")
				if not x_value:
					continue
			iterations.append(int(x_value))
			best_values.append(float(value))
	return iterations, best_values


def _dedupe_x(iterations: list[int], values: list[float]) -> tuple[list[int], list[float]]:
	by_x = {}
	for iteration, value in zip(iterations, values):
		by_x[iteration] = value
	return list(by_x), [by_x[key] for key in by_x]


def _load_counts_by_exp_id(db_path: Path) -> tuple[dict[int, dict[str, int]], list[str]]:
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	try:
		rows = conn.execute(
			"""
			SELECT exp_id, total_instruction_counts_json
			FROM llvm_instruction_counts
			ORDER BY exp_id ASC
			"""
		).fetchall()
	finally:
		conn.close()

	grouped: dict[int, dict[str, int]] = defaultdict(dict)
	all_ops: set[str] = set()
	for row in rows:
		exp_id = int(row["exp_id"])
		counts = json.loads(row["total_instruction_counts_json"])
		dst = grouped.setdefault(exp_id, {})
		for op, value in counts.items():
			dst[op] = dst.get(op, 0) + int(value)
			all_ops.add(str(op))
	return dict(grouped), filter_instruction_opcodes(all_ops)


def _euclidean_distance(a: dict[str, int], b: dict[str, int], opcodes: list[str]) -> float:
	return math.sqrt(
		sum((float(a.get(op, 0)) - float(b.get(op, 0))) ** 2 for op in opcodes)
	)


def _load_new_best_distance_series(method_dir: Path) -> tuple[list[int], list[float]]:
	trace_path = method_dir / "trace.csv"
	db_path = method_dir / "live_experiments.db"
	if not trace_path.exists() or not db_path.exists():
		return [], []
	counts_by_exp_id, opcodes = _load_counts_by_exp_id(db_path)
	if not opcodes:
		return [], []

	best_event = 0
	last_best_runtime: float | None = None
	last_best_exp_id: int | None = None
	x_values: list[int] = []
	distances: list[float] = []
	with trace_path.open() as f:
		for row in csv.DictReader(f):
			best_runtime_text = row.get("best_runtime_ms", "")
			exp_id_text = row.get("exp_id", "")
			if not best_runtime_text or not exp_id_text:
				continue
			best_runtime = float(best_runtime_text)
			exp_id = int(exp_id_text)
			if last_best_runtime is not None and best_runtime >= last_best_runtime:
				continue
			best_event += 1
			if last_best_exp_id is not None:
				prev_counts = counts_by_exp_id.get(last_best_exp_id)
				cur_counts = counts_by_exp_id.get(exp_id)
				if prev_counts is not None and cur_counts is not None:
					x_values.append(best_event)
					distances.append(_euclidean_distance(prev_counts, cur_counts, opcodes))
			last_best_runtime = best_runtime
			last_best_exp_id = exp_id
	return x_values, distances


def plot_run(
	run_dir: Path,
	output_path: Path,
	title: str | None = None,
	x_axis: str = "iteration",
	common_x_limit: bool = False,
	x_limit: int | None = None,
) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot convergence graphs") from exc

	series = {
		"parameter tuning": _load_series(run_dir / "parameter_tuning" / "trace.csv", x_axis),
		"instruction-map tuning": _load_series(run_dir / "instruction_map_tuning" / "trace.csv", x_axis),
	}
	if common_x_limit:
		max_x_values = [max(iterations) for iterations, _ in series.values() if iterations]
		if max_x_values:
			limit = min(max_x_values)
			if x_limit is not None:
				limit = min(limit, x_limit)
			series = {
				label: (
					[x for x, _ in zip(iterations, values) if x <= limit],
					[y for x, y in zip(iterations, values) if x <= limit],
				)
				for label, (iterations, values) in series.items()
			}
	elif x_limit is not None:
		series = {
			label: (
				[x for x, _ in zip(iterations, values) if x <= x_limit],
				[y for x, y in zip(iterations, values) if x <= x_limit],
			)
			for label, (iterations, values) in series.items()
		}

	fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
	ax.set_title(title or run_dir.name)

	for label, (iterations, best_values) in series.items():
		if not iterations:
			continue
		if x_axis in {"valid", "unique_valid"}:
			iterations, best_values = _dedupe_x(iterations, best_values)
		ax.plot(iterations, best_values, label=label, linewidth=2)

	ax.set_ylabel("Best runtime so far (ms)")
	ax.set_yscale("log")
	ax.grid(True, alpha=0.3)
	ax.legend()

	xlabel = {
		"iteration": "OpenTuner attempt",
		"valid": "Valid dataset evaluation",
		"unique_valid": "Unique valid dataset configuration",
	}[x_axis]
	ax.set_xlabel(xlabel)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def plot_new_best_distance_run(
	run_dir: Path,
	output_path: Path,
	title: str | None = None,
) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot distance graphs") from exc

	series = {
		"parameter tuning": _load_new_best_distance_series(run_dir / "parameter_tuning"),
		"instruction-map tuning": _load_new_best_distance_series(run_dir / "instruction_map_tuning"),
	}

	fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
	ax.set_title(title or run_dir.name)

	for label, (x_values, distances) in series.items():
		if not x_values:
			continue
		ax.plot(x_values, distances, label=label, linewidth=2)

	ax.set_xlabel("New best event")
	ax.set_ylabel("Instruction-map distance to previous best")
	ax.set_yscale("log")
	ax.grid(True, alpha=0.3)
	ax.legend()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def main() -> int:
	parser = argparse.ArgumentParser(description="Plot convergence comparison from two tuner traces")
	parser.add_argument("run_dir", help="Directory containing parameter_tuning/ and instruction_map_tuning/")
	parser.add_argument("--output", default=None, help="Output image path")
	parser.add_argument("--title", default=None)
	parser.add_argument(
		"--x-axis",
		choices=["iteration", "valid", "unique_valid"],
		default="iteration",
		help="Use all OpenTuner attempts, valid evaluations, or unique valid configs as the x-axis.",
	)
	parser.add_argument(
		"--common-x-limit",
		action="store_true",
		help="Truncate all curves to the smallest maximum x-value among methods.",
	)
	parser.add_argument("--x-limit", type=int, default=None, help="Optional numeric x-axis cutoff.")
	args = parser.parse_args()

	run_dir = Path(args.run_dir)
	output = Path(args.output) if args.output else run_dir / "convergence.png"
	plot_run(
		run_dir,
		output,
		title=args.title,
		x_axis=args.x_axis,
		common_x_limit=args.common_x_limit,
		x_limit=args.x_limit,
	)
	print(output)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
