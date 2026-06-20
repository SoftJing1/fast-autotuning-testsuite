from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


CASES = (
	("gaussian_512", "Gaussian 512x512"),
	("gaussian_1024", "Gaussian 1024x1024"),
	("gaussian_2048", "Gaussian 2048x2048"),
	("gemm_128", "GEMM 128x128x128"),
	("gemm_256", "GEMM 256x256x256"),
	("gemm_512", "GEMM 512x512x512"),
)

MILESTONES = (10, 25, 50, 100, 200, 300)


@dataclass(frozen=True)
class Experiment:
	label: str
	root: Path
	method: str = "instruction_map_tuning"


def _safe_float(value: Any, default: float = math.nan) -> float:
	if value in (None, ""):
		return default
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _safe_int(value: Any, default: int = 0) -> int:
	if value in (None, ""):
		return default
	try:
		return int(value)
	except (TypeError, ValueError):
		return default


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


def _format_float(value: float) -> str:
	if not math.isfinite(value):
		return "n/a"
	return f"{value:.6f}"


def _format_ratio(value: float) -> str:
	if not math.isfinite(value):
		return "n/a"
	return f"{value:.3f}x"


def _relpath(path: Path, output_dir: Path) -> str:
	return os.path.relpath(path.resolve(), output_dir.resolve())


def _method_dir(experiment: Experiment, case_name: str, seed: int) -> Path:
	return experiment.root / case_name / f"seed_{seed}" / experiment.method


def _load_trace(trace_path: Path) -> list[dict[str, str]]:
	if not trace_path.exists():
		return []
	with trace_path.open(newline="") as f:
		return list(csv.DictReader(f))


def _load_run_config(method_dir: Path) -> dict[str, Any]:
	path = method_dir / "run_config.json"
	if not path.exists():
		return {}
	return json.loads(path.read_text())


def _load_summary(method_dir: Path) -> dict[str, Any]:
	path = method_dir / "summary.json"
	if path.exists():
		return json.loads(path.read_text())
	trace = _load_trace(method_dir / "trace.csv")
	if not trace:
		return {}
	last = trace[-1]
	return {
		"best_runtime_ms": _safe_float(last.get("best_runtime_ms")),
		"unique_config_count": _safe_int(last.get("unique_config_count")),
		"valid_evaluation_count": _safe_int(last.get("valid_evaluation_count")),
		"iterations": _safe_int(last.get("iteration")),
	}


def _best_series(trace_path: Path, valid_limit: int | None = None) -> dict[int, float]:
	series: dict[int, float] = {}
	for row in _load_trace(trace_path):
		x = _safe_int(row.get("valid_evaluation_count"))
		y = _safe_float(row.get("best_runtime_ms"))
		if valid_limit is not None and x > valid_limit:
			continue
		if x > 0 and math.isfinite(y):
			series[x] = y
	return series


def _milestone_values(trace_path: Path, milestones: tuple[int, ...], valid_limit: int | None) -> dict[int, float]:
	series = _best_series(trace_path, valid_limit=valid_limit)
	values: dict[int, float] = {}
	if not series:
		return values
	max_x = max(series)
	for target in milestones:
		if target > max_x:
			continue
		available = [x for x in series if x >= target]
		if available:
			values[target] = series[min(available)]
	return values


def _trace_diagnostics(trace_path: Path, valid_limit: int | None) -> dict[str, float]:
	db_distances: list[float] = []
	refined_distances: list[float] = []
	distance_ratios: list[float] = []
	refinement_times: list[float] = []
	for row in _load_trace(trace_path):
		valid = _safe_int(row.get("valid_evaluation_count"))
		if valid <= 0:
			continue
		if valid_limit is not None and valid > valid_limit:
			break
		db_distance = _safe_float(row.get("db_resolver_distance"))
		refined_distance = _safe_float(row.get("refined_resolver_distance"))
		distance_ratio = _safe_float(row.get("distance_ratio"))
		refinement_time = _safe_float(row.get("resolver_refinement_time_ms"))
		if math.isfinite(db_distance):
			db_distances.append(db_distance)
		if math.isfinite(refined_distance):
			refined_distances.append(refined_distance)
		if math.isfinite(distance_ratio):
			distance_ratios.append(distance_ratio)
		if math.isfinite(refinement_time):
			refinement_times.append(refinement_time)
	return {
		"median_db_resolver_distance": median(db_distances) if db_distances else math.nan,
		"median_refined_resolver_distance": median(refined_distances) if refined_distances else math.nan,
		"median_distance_ratio": median(distance_ratios) if distance_ratios else math.nan,
		"target_10pct_hit_rate": (
			sum(1 for value in distance_ratios if value <= 0.1) / len(distance_ratios) * 100.0
			if distance_ratios
			else math.nan
		),
		"median_resolver_refinement_time_ms": median(refinement_times) if refinement_times else math.nan,
	}


def _seed_numbers(experiments: list[Experiment], case_name: str) -> list[int]:
	seeds: set[int] = set()
	for experiment in experiments:
		case_dir = experiment.root / case_name
		for seed_dir in case_dir.glob("seed_*"):
			try:
				seed = int(seed_dir.name.split("_", 1)[1])
			except (IndexError, ValueError):
				continue
			if (seed_dir / experiment.method / "trace.csv").exists():
				seeds.add(seed)
	return sorted(seeds)


def _aggregate(
	experiment: Experiment,
	case_name: str,
	valid_limit: int | None,
	milestones: tuple[int, ...],
) -> dict[str, Any]:
	seed_series: list[dict[int, float]] = []
	best_values: list[float] = []
	unique_values: list[int] = []
	valid_values: list[int] = []
	iteration_values: list[int] = []
	seed_rows: list[dict[str, Any]] = []
	run_config: dict[str, Any] = {}

	for seed in _seed_numbers([experiment], case_name):
		method_dir = _method_dir(experiment, case_name, seed)
		if not run_config:
			run_config = _load_run_config(method_dir)
		trace_path = method_dir / "trace.csv"
		summary = _summary_at_limit(method_dir, valid_limit)
		series = _best_series(trace_path, valid_limit=valid_limit)
		if series:
			seed_series.append(series)
		best = _safe_float(summary.get("best_runtime_ms"))
		unique = _safe_int(summary.get("unique_config_count"))
		valid = _safe_int(summary.get("valid_evaluation_count"))
		iterations = _safe_int(summary.get("iterations"))
		if math.isfinite(best):
			best_values.append(best)
		if unique:
			unique_values.append(unique)
		if valid:
			valid_values.append(valid)
		if iterations:
			iteration_values.append(iterations)
		seed_rows.append(
			{
				"seed": seed,
				"best_runtime_ms": best,
				"unique_config_count": unique,
				"valid_evaluation_count": valid,
				"iterations": iterations,
				"milestones": _milestone_values(trace_path, milestones, valid_limit),
				"diagnostics": _trace_diagnostics(trace_path, valid_limit),
				"trace_path": str(trace_path),
			}
		)

	x_values: list[int] = []
	medians: list[float] = []
	q1_values: list[float] = []
	q3_values: list[float] = []
	max_x = max((max(series) for series in seed_series), default=0)
	for x in range(1, max_x + 1):
		values = sorted(series[x] for series in seed_series if x in series)
		if not values:
			continue
		x_values.append(x)
		medians.append(median(values))
		q1_values.append(_quantile(values, 0.25))
		q3_values.append(_quantile(values, 0.75))

	return {
		"seed_count": len(seed_rows),
		"x": x_values,
		"median": medians,
		"q1": q1_values,
		"q3": q3_values,
		"median_best_runtime_ms": median(best_values) if best_values else math.nan,
		"best_runtime_ms_min": min(best_values) if best_values else math.nan,
		"best_runtime_ms_max": max(best_values) if best_values else math.nan,
		"median_unique_config_count": median(unique_values) if unique_values else 0,
		"median_valid_evaluation_count": median(valid_values) if valid_values else 0,
		"median_iterations": median(iteration_values) if iteration_values else 0,
		"diagnostics": _aggregate_diagnostics(seed_rows),
		"seed_rows": seed_rows,
		"run_config": {
			"distance_metric": run_config.get("distance_metric"),
			"method": experiment.method,
			"resolver_refinement_mode": run_config.get("resolver_refinement_mode"),
			"resolver_inner_valid_profile_limit": run_config.get("resolver_inner_valid_profile_limit"),
			"warmup_runs": run_config.get("warmup_runs"),
			"runs_per_config": run_config.get("runs_per_config"),
			"max_values_per_op": run_config.get("max_values_per_op"),
			"resolver_db": run_config.get("resolver_db"),
		},
	}


def _aggregate_diagnostics(seed_rows: list[dict[str, Any]]) -> dict[str, float]:
	keys = (
		"median_db_resolver_distance",
		"median_refined_resolver_distance",
		"median_distance_ratio",
		"target_10pct_hit_rate",
		"median_resolver_refinement_time_ms",
	)
	result: dict[str, float] = {}
	for key in keys:
		values = sorted(
			float(row["diagnostics"][key])
			for row in seed_rows
			if math.isfinite(float(row["diagnostics"][key]))
		)
		result[key] = median(values) if values else math.nan
	return result


def _summary_at_limit(method_dir: Path, valid_limit: int | None) -> dict[str, Any]:
	if valid_limit is None:
		return _load_summary(method_dir)
	trace = _load_trace(method_dir / "trace.csv")
	if not trace:
		return {}
	last = None
	for row in trace:
		valid = _safe_int(row.get("valid_evaluation_count"))
		if valid > valid_limit:
			break
		if valid > 0:
			last = row
	if last is None:
		return {}
	return {
		"best_runtime_ms": _safe_float(last.get("best_runtime_ms")),
		"unique_config_count": _safe_int(last.get("unique_config_count")),
		"valid_evaluation_count": _safe_int(last.get("valid_evaluation_count")),
		"iterations": _safe_int(last.get("iteration")),
	}


def _plot_case(
	case_title: str,
	aggregates: dict[str, dict[str, Any]],
	output_path: Path,
) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot comparison graphs") from exc

	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
	fig, ax = plt.subplots(1, 1, figsize=(9.4, 4.8))
	ax.set_title(case_title)
	for index, (label, aggregate) in enumerate(aggregates.items()):
		if not aggregate["x"]:
			continue
		color = colors[index % len(colors)]
		ax.fill_between(aggregate["x"], aggregate["q1"], aggregate["q3"], color=color, alpha=0.18)
		ax.plot(aggregate["x"], aggregate["median"], label=label, linewidth=2, color=color)
	ax.set_xlabel("Valid evaluation")
	ax.set_ylabel("Best runtime so far (ms)")
	ax.set_yscale("log")
	ax.grid(True, alpha=0.3)
	ax.legend()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def _plot_seed(
	case_title: str,
	seed: int,
	experiments: list[Experiment],
	case_name: str,
	output_path: Path,
	valid_limit: int | None,
) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot comparison graphs") from exc

	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
	fig, ax = plt.subplots(1, 1, figsize=(9.4, 4.8))
	ax.set_title(f"{case_title} seed {seed}")
	for index, experiment in enumerate(experiments):
		series = _best_series(
			_method_dir(experiment, case_name, seed) / "trace.csv",
			valid_limit=valid_limit,
		)
		if not series:
			continue
		x = sorted(series)
		y = [series[value] for value in x]
		ax.plot(x, y, label=experiment.label, linewidth=2, color=colors[index % len(colors)])
	ax.set_xlabel("Valid evaluation")
	ax.set_ylabel("Best runtime so far (ms)")
	ax.set_yscale("log")
	ax.grid(True, alpha=0.3)
	ax.legend()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def _aggregate_rows(experiments: list[Experiment], aggregates: dict[str, dict[str, Any]]) -> str:
	rows = []
	for experiment in experiments:
		aggregate = aggregates[experiment.label]
		config = aggregate["run_config"]
		diagnostics = aggregate["diagnostics"]
		rows.append(
			"<tr>"
			f"<td>{html.escape(experiment.label)}</td>"
			f"<td>{html.escape(str(config.get('method') or experiment.method))}</td>"
			f"<td>{html.escape(str(config.get('distance_metric') or ''))}</td>"
			f"<td>{html.escape(str(config.get('resolver_refinement_mode') or ''))}</td>"
			f"<td>{html.escape(str(config.get('resolver_inner_valid_profile_limit') or ''))}</td>"
			f"<td>{html.escape(str(config.get('warmup_runs') or ''))}</td>"
			f"<td>{html.escape(str(config.get('runs_per_config') or ''))}</td>"
			f"<td>{aggregate['seed_count']}</td>"
			f"<td>{_format_float(aggregate['median_best_runtime_ms'])}</td>"
			f"<td>{_format_float(aggregate['best_runtime_ms_min'])}</td>"
			f"<td>{_format_float(aggregate['best_runtime_ms_max'])}</td>"
			f"<td>{aggregate['median_unique_config_count']}</td>"
			f"<td>{aggregate['median_valid_evaluation_count']}</td>"
			f"<td>{aggregate['median_iterations']}</td>"
			f"<td>{_format_float(diagnostics['median_db_resolver_distance'])}</td>"
			f"<td>{_format_float(diagnostics['median_refined_resolver_distance'])}</td>"
			f"<td>{_format_ratio(diagnostics['median_distance_ratio'])}</td>"
			f"<td>{_format_float(diagnostics['target_10pct_hit_rate'])}%</td>"
			f"<td>{_format_float(diagnostics['median_resolver_refinement_time_ms'])}</td>"
			"</tr>"
		)
	return "".join(rows)


def _seed_rows(
	experiments: list[Experiment],
	aggregates: dict[str, dict[str, Any]],
	output_dir: Path,
	milestones: tuple[int, ...],
) -> str:
	seed_set = sorted(
		{
			int(row["seed"])
			for aggregate in aggregates.values()
			for row in aggregate["seed_rows"]
		}
	)
	by_label_seed = {
		(label, int(row["seed"])): row
		for label, aggregate in aggregates.items()
		for row in aggregate["seed_rows"]
	}
	rows: list[str] = []
	base_label = experiments[0].label if experiments else ""
	for seed in seed_set:
		base_row = by_label_seed.get((base_label, seed))
		for experiment in experiments:
			row = by_label_seed.get((experiment.label, seed))
			if row is None:
				continue
			ratio = math.nan
			if base_row is not None and math.isfinite(base_row["best_runtime_ms"]) and base_row["best_runtime_ms"] > 0:
				ratio = row["best_runtime_ms"] / base_row["best_runtime_ms"]
			row_milestones = row["milestones"]
			diagnostics = row["diagnostics"]
			trace_link = _relpath(Path(row["trace_path"]), output_dir)
			rows.append(
				"<tr>"
				f"<td>{seed}</td>"
				f"<td>{html.escape(experiment.label)}</td>"
				f"<td>{_format_float(row['best_runtime_ms'])}</td>"
				f"<td>{_format_ratio(ratio)}</td>"
				f"<td>{row['unique_config_count']}</td>"
				f"<td>{row['valid_evaluation_count']}</td>"
				f"<td>{row['iterations']}</td>"
				f"<td>{_format_ratio(diagnostics['median_distance_ratio'])}</td>"
				f"<td>{_format_float(diagnostics['target_10pct_hit_rate'])}%</td>"
				+ "".join(f"<td>{_format_float(float(row_milestones.get(target, math.nan)))}</td>" for target in milestones)
				+ f"<td><a href=\"{html.escape(trace_link)}\">trace</a></td>"
				"</tr>"
			)
	return "".join(rows)


def build_report(experiments: list[Experiment], output_dir: Path, valid_limit: int | None = None) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	milestones = tuple(target for target in MILESTONES if valid_limit is None or target <= valid_limit)
	report_data: dict[str, Any] = {}
	sections: list[str] = []
	for case_name, case_title in CASES:
		aggregates = {
			experiment.label: _aggregate(experiment, case_name, valid_limit, milestones)
			for experiment in experiments
		}
		report_data[case_name] = aggregates
		case_plot = output_dir / case_name / "aggregate.png"
		_plot_case(case_title, aggregates, case_plot)
		seed_figures: list[str] = []
		for seed in _seed_numbers(experiments, case_name):
			seed_plot = output_dir / case_name / f"seed_{seed}.png"
			_plot_seed(case_title, seed, experiments, case_name, seed_plot, valid_limit)
			seed_figures.append(
				f"""
				<div class="seed-plot">
					<h3>Seed {seed}</h3>
					<img src="{html.escape(_relpath(seed_plot, output_dir))}" alt="{html.escape(case_title)} seed {seed}">
				</div>
				"""
			)
		sections.append(
			f"""
			<section>
				<h2>{html.escape(case_title)}</h2>
				<table>
					<thead>
						<tr>
							<th>Experiment</th><th>Method</th><th>Metric</th><th>Refine</th><th>Inner valid</th><th>Warmup</th><th>Runs</th>
							<th>Seeds</th><th>Median best</th><th>Best min</th><th>Best max</th>
							<th>Median unique</th><th>Median valid</th><th>Median iterations</th>
							<th>Median DB dist</th><th>Median refined dist</th><th>Median dist ratio</th>
							<th>&lt;=10% dist</th><th>Median refine ms</th>
						</tr>
					</thead>
					<tbody>{_aggregate_rows(experiments, aggregates)}</tbody>
				</table>
				<img src="{html.escape(_relpath(case_plot, output_dir))}" alt="{html.escape(case_title)} aggregate comparison">
				<h3>Per-Seed Details</h3>
				<table>
					<thead>
						<tr>
							<th>Seed</th><th>Experiment</th><th>Best runtime</th><th>Ratio to {html.escape(experiments[0].label)}</th>
							<th>Unique</th><th>Valid</th><th>Iterations</th>
							<th>Median dist ratio</th><th>&lt;=10% dist</th>
							{''.join(f'<th>Best@{target}</th>' for target in milestones)}
							<th>Trace</th>
						</tr>
					</thead>
					<tbody>{_seed_rows(experiments, aggregates, output_dir, milestones)}</tbody>
				</table>
				<div class="seed-grid">{''.join(seed_figures)}</div>
			</section>
			"""
		)

	(output_dir / "summary.json").write_text(
		json.dumps(report_data, indent=2, sort_keys=True),
		encoding="utf-8",
	)
	(output_dir / "index.html").write_text(
		f"""<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Instruction Distance Metric Comparison</title>
	<style>
		body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f8fa; }}
		header {{ padding: 28px 36px; background: #1f2933; color: white; }}
		main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
		section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
		h1, h2 {{ margin: 0 0 14px; }}
		h3 {{ margin: 18px 0 8px; font-size: 16px; }}
		p {{ margin: 4px 0 0; color: #bcccdc; }}
		table {{ width: 100%; border-collapse: collapse; margin: 14px 0 18px; font-size: 13px; }}
		th, td {{ text-align: right; padding: 7px 8px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }}
		th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
		img {{ width: 100%; max-width: 980px; display: block; margin: 8px auto 16px; border: 1px solid #e5e7eb; }}
		a {{ color: #0969da; text-decoration: none; }}
		a:hover {{ text-decoration: underline; }}
		.seed-grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
		.seed-plot {{ border-top: 1px solid #e5e7eb; padding-top: 10px; }}
		.note {{ color: #d9e2ec; max-width: 940px; }}
	</style>
</head>
<body>
	<header>
		<h1>Instruction Distance Metric Comparison</h1>
		<p class="note">Aggregate and per-seed comparison of tuning runs. X-axis is valid evaluations{'' if valid_limit is None else f' up to {valid_limit}'}; Y-axis is best runtime so far on a log scale.</p>
	</header>
	<main>{''.join(sections)}</main>
</body>
</html>
""",
		encoding="utf-8",
	)


def _parse_experiment(value: str) -> Experiment:
	if "=" not in value:
		raise argparse.ArgumentTypeError("experiment must be formatted as label=path")
	label, path = value.split("=", 1)
	if not label or not path:
		raise argparse.ArgumentTypeError("experiment must be formatted as label=path")
	method = "instruction_map_tuning"
	for suffix in (":instruction_map_tuning", ":parameter_tuning"):
		if path.endswith(suffix):
			path = path[: -len(suffix)]
			method = suffix[1:]
			break
	return Experiment(label=label, root=Path(path), method=method)


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate an HTML report comparing instruction distance metrics.")
	parser.add_argument("--experiment", action="append", type=_parse_experiment, required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--valid-limit", type=int, default=None)
	args = parser.parse_args()
	build_report(args.experiment, Path(args.output_dir), valid_limit=args.valid_limit)
	print(Path(args.output_dir) / "index.html")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
