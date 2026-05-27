from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import median

from .plot_comparison import plot_run


CASES = (
	("gaussian_512", "Gaussian 512x512"),
	("gaussian_1024", "Gaussian 1024x1024"),
	("gaussian_2048", "Gaussian 2048x2048"),
	("gemm_128", "GEMM 128x128x128"),
	("gemm_256", "GEMM 256x256x256"),
	("gemm_512", "GEMM 512x512x512"),
)

METHODS = (
	("parameter_tuning", "Parameter tuning"),
	("instruction_map_tuning", "Instruction-map tuning"),
)


def _quantile(sorted_values: list[float], q: float) -> float:
	if not sorted_values:
		return math.nan
	if len(sorted_values) == 1:
		return sorted_values[0]
	pos = (len(sorted_values) - 1) * q
	lower = math.floor(pos)
	upper = math.ceil(pos)
	if lower == upper:
		return sorted_values[lower]
	weight = pos - lower
	return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _load_unique_valid_series(trace_path: Path) -> dict[int, float]:
	series: dict[int, float] = {}
	with trace_path.open() as f:
		for row in csv.DictReader(f):
			x_value = row.get("unique_config_count", "")
			y_value = row.get("best_runtime_ms", "")
			if not x_value or not y_value:
				continue
			x = int(x_value)
			if x <= 0:
				continue
			series[x] = float(y_value)
	return series


def _seed_method_dirs(case_dir: Path, method_dir_name: str) -> list[Path]:
	return sorted(
		path / method_dir_name
		for path in case_dir.glob("seed_*")
		if (path / method_dir_name / "trace.csv").exists()
	)


def _seed_numbers(case_dir: Path) -> list[int]:
	seeds: list[int] = []
	for path in sorted(case_dir.glob("seed_*")):
		if not path.is_dir():
			continue
		try:
			seeds.append(int(path.name.split("_", 1)[1]))
		except (IndexError, ValueError):
			continue
	return seeds


def _aggregate_method(case_dir: Path, method_dir_name: str) -> dict:
	seed_dirs = _seed_method_dirs(case_dir, method_dir_name)
	seed_series: list[dict[int, float]] = []
	best_values: list[float] = []
	final_unique_counts: list[int] = []
	final_valid_counts: list[int] = []
	final_iterations: list[int] = []

	for method_dir in seed_dirs:
		trace_path = method_dir / "trace.csv"
		summary_path = method_dir / "summary.json"
		series = _load_unique_valid_series(trace_path)
		if series:
			seed_series.append(series)
		if summary_path.exists():
			summary = json.loads(summary_path.read_text())
			best_values.append(float(summary.get("best_runtime_ms", math.nan)))
			final_unique_counts.append(int(summary.get("unique_config_count", 0)))
			final_valid_counts.append(int(summary.get("valid_evaluation_count", 0)))
			final_iterations.append(int(summary.get("iterations", 0)))

	max_x = max((max(series) for series in seed_series), default=0)
	x_values: list[int] = []
	medians: list[float] = []
	q1_values: list[float] = []
	q3_values: list[float] = []
	counts: list[int] = []

	for x in range(1, max_x + 1):
		values = sorted(series[x] for series in seed_series if x in series)
		if not values:
			continue
		x_values.append(x)
		medians.append(median(values))
		q1_values.append(_quantile(values, 0.25))
		q3_values.append(_quantile(values, 0.75))
		counts.append(len(values))

	finite_best_values = [value for value in best_values if math.isfinite(value)]
	return {
		"seed_count": len(seed_dirs),
		"x": x_values,
		"median": medians,
		"q1": q1_values,
		"q3": q3_values,
		"counts": counts,
		"median_best_runtime_ms": median(finite_best_values) if finite_best_values else math.nan,
		"best_runtime_ms_min": min(finite_best_values) if finite_best_values else math.nan,
		"best_runtime_ms_max": max(finite_best_values) if finite_best_values else math.nan,
		"median_final_unique_configs": median(final_unique_counts) if final_unique_counts else 0,
		"median_final_valid_evaluations": median(final_valid_counts) if final_valid_counts else 0,
		"median_final_iterations": median(final_iterations) if final_iterations else 0,
	}


def plot_case_aggregate(case_dir: Path, output_path: Path, title: str) -> dict[str, dict]:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot aggregate convergence graphs") from exc

	aggregates = {
		method_dir_name: _aggregate_method(case_dir, method_dir_name)
		for method_dir_name, _ in METHODS
	}

	fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
	ax.set_title(title)
	colors = {
		"parameter_tuning": "#1f77b4",
		"instruction_map_tuning": "#d62728",
	}

	for method_dir_name, label in METHODS:
		aggregate = aggregates[method_dir_name]
		if not aggregate["x"]:
			continue
		color = colors[method_dir_name]
		ax.plot(aggregate["x"], aggregate["median"], label=label, linewidth=2, color=color)
		ax.fill_between(
			aggregate["x"],
			aggregate["q1"],
			aggregate["q3"],
			color=color,
			alpha=0.18,
			linewidth=0,
		)

	ax.set_xlabel("Unique valid configuration")
	ax.set_ylabel("Best runtime so far (ms)")
	ax.set_yscale("log")
	ax.grid(True, alpha=0.3)
	ax.legend()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)
	return aggregates


def _format_float(value: float) -> str:
	if not math.isfinite(value):
		return "n/a"
	return f"{value:.6f}"


def build_report(run_root: Path, output: Path) -> None:
	sections: list[str] = []
	report_data: dict[str, dict[str, dict]] = {}

	for case_dir_name, title in CASES:
		case_dir = run_root / case_dir_name
		plot_path = case_dir / "aggregate_convergence_unique_valid.png"
		aggregates = plot_case_aggregate(case_dir, plot_path, title)
		report_data[case_dir_name] = aggregates
		plot_rel = plot_path.relative_to(output.parent)
		seed_figure_blocks: list[str] = []
		case_seeds = _seed_numbers(case_dir)
		for seed in case_seeds:
			seed_dir = case_dir / f"seed_{seed}"
			if not seed_dir.exists():
				continue
			seed_plot_path = seed_dir / "convergence_unique_valid.png"
			plot_run(
				seed_dir,
				seed_plot_path,
				title=f"{title} - Seed {seed}",
				x_axis="unique_valid",
			)
			seed_plot_rel = seed_plot_path.relative_to(output.parent)
			seed_figure_blocks.append(
				f"""
				<div class="seed-figure">
					<h3>Seed {seed}</h3>
					<img src="{html.escape(str(seed_plot_rel))}" alt="{html.escape(title)} seed {seed} convergence plot">
				</div>
				"""
			)

		rows = []
		for method_dir_name, label in METHODS:
			aggregate = aggregates[method_dir_name]
			rows.append(
				"<tr>"
				f"<td>{html.escape(label)}</td>"
				f"<td>{aggregate['seed_count']}</td>"
				f"<td>{aggregate['median_final_unique_configs']}</td>"
				f"<td>{aggregate['median_final_valid_evaluations']}</td>"
				f"<td>{aggregate['median_final_iterations']}</td>"
				f"<td>{_format_float(aggregate['median_best_runtime_ms'])}</td>"
				f"<td>{_format_float(aggregate['best_runtime_ms_min'])}</td>"
				f"<td>{_format_float(aggregate['best_runtime_ms_max'])}</td>"
				"</tr>"
			)

		sections.append(
			f"""
			<section>
				<h2>{html.escape(title)}</h2>
				<table>
					<thead>
						<tr>
							<th>Method</th>
							<th>Seeds</th>
							<th>Median final unique</th>
							<th>Median final valid evals</th>
							<th>Median final iterations</th>
							<th>Median best runtime (ms)</th>
							<th>Best runtime min</th>
							<th>Best runtime max</th>
						</tr>
					</thead>
					<tbody>
						{''.join(rows)}
					</tbody>
				</table>
				<img src="{html.escape(str(plot_rel))}" alt="{html.escape(title)} aggregate convergence plot">
				<p class="caption">Line: median over {aggregates['parameter_tuning']['seed_count']} seeds. Shaded band: interquartile range.</p>
				<div class="seed-grid">
					{''.join(seed_figure_blocks)}
				</div>
			</section>
			"""
		)

	output.write_text(
		f"""<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Two-Step Autotuning Aggregate Report</title>
	<style>
		body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f8fa; }}
		header {{ padding: 28px 36px; background: #1f2933; color: white; }}
		main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
		section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
		h1, h2 {{ margin: 0 0 14px; }}
		h3 {{ margin: 0 0 8px; font-size: 15px; }}
		p {{ margin: 4px 0 0; color: #bcccdc; }}
		p.caption {{ color: #52606d; margin-top: 8px; }}
		table {{ width: 100%; border-collapse: collapse; margin: 14px 0 18px; font-size: 14px; }}
		th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
		th:first-child, td:first-child {{ text-align: left; }}
		img {{ width: 100%; max-width: 960px; display: block; margin: 8px auto 12px; border: 1px solid #e5e7eb; }}
		.seed-grid {{ display: grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
		.seed-figure {{ border-top: 1px solid #e5e7eb; padding-top: 12px; }}
	</style>
</head>
<body>
	<header>
		<h1>Two-Step Autotuning Aggregate Report</h1>
		<p>Aggregate over all recorded seeds per case. X-axis is unique valid configurations. Y-axis is best runtime so far, log scale.</p>
	</header>
	<main>
		{''.join(sections)}
	</main>
</body>
</html>
""",
		encoding="utf-8",
	)

	(run_root / "aggregate_summary.json").write_text(
		json.dumps(report_data, indent=2, sort_keys=True),
		encoding="utf-8",
	)


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate aggregate convergence report over replicate autotuning runs.")
	parser.add_argument("run_root")
	parser.add_argument("--output", default=None)
	args = parser.parse_args()

	run_root = Path(args.run_root)
	output = Path(args.output) if args.output else run_root / "index_aggregate.html"
	build_report(run_root, output)
	print(output)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
