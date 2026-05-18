from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
