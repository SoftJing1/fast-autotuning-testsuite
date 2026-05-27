from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .aggregate_replicate_report import CASES, _aggregate_method, _format_float, _seed_numbers
from .plot_comparison import _dedupe_x, _load_series


def plot_case_compare(
	case_dir_name: str,
	title: str,
	experiments: list[tuple[str, Path]],
	output_path: Path,
) -> dict[str, dict]:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot comparison graphs") from exc

	aggregates: dict[str, dict] = {}
	for label, run_root in experiments:
		aggregates[label] = _aggregate_method(run_root / case_dir_name, "parameter_tuning")

	fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
	ax.set_title(title)
	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

	for index, (label, _) in enumerate(experiments):
		aggregate = aggregates[label]
		if not aggregate["x"]:
			continue
		color = colors[index % len(colors)]
		ax.fill_between(
			aggregate["x"],
			aggregate["q1"],
			aggregate["q3"],
			color=color,
			alpha=0.18,
			linewidth=0,
		)
		ax.plot(
			aggregate["x"],
			aggregate["median"],
			label=label,
			linewidth=2,
			color=color,
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


def plot_seed_compare(
	case_dir_name: str,
	seed: int,
	title: str,
	experiments: list[tuple[str, Path]],
	output_path: Path,
) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot comparison graphs") from exc

	fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
	ax.set_title(title)
	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

	for index, (label, run_root) in enumerate(experiments):
		trace_path = run_root / case_dir_name / f"seed_{seed}" / "parameter_tuning" / "trace.csv"
		if not trace_path.exists():
			continue
		x_values, y_values = _load_series(trace_path, "unique_valid")
		if not x_values:
			continue
		x_values, y_values = _dedupe_x(x_values, y_values)
		ax.plot(
			x_values,
			y_values,
			label=label,
			linewidth=2,
			color=colors[index % len(colors)],
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


def build_report(
	experiments: list[tuple[str, Path]],
	output_dir: Path,
) -> None:
	sections: list[str] = []
	report_data: dict[str, dict[str, dict]] = {}

	for case_dir_name, title in CASES:
		plot_path = output_dir / case_dir_name / "parameter_compare.png"
		aggregates = plot_case_compare(case_dir_name, title, experiments, plot_path)
		report_data[case_dir_name] = aggregates
		plot_rel = plot_path.relative_to(output_dir)
		seed_figure_blocks: list[str] = []
		case_seeds: set[int] = set()
		for _, run_root in experiments:
			case_seeds.update(_seed_numbers(run_root / case_dir_name))
		for seed in sorted(case_seeds):
			seed_plot_path = output_dir / case_dir_name / f"seed_{seed}" / "parameter_compare.png"
			plot_seed_compare(
				case_dir_name,
				seed,
				f"{title} - Seed {seed}",
				experiments,
				seed_plot_path,
			)
			seed_plot_rel = seed_plot_path.relative_to(output_dir)
			seed_figure_blocks.append(
				f"""
				<div class="seed-figure">
					<h3>Seed {seed}</h3>
					<img src="{html.escape(str(seed_plot_rel))}" alt="{html.escape(title)} seed {seed} parameter experiment comparison">
				</div>
				"""
			)

		rows = []
		for label, _ in experiments:
			aggregate = aggregates[label]
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
							<th>Experiment</th>
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
				<img src="{html.escape(str(plot_rel))}" alt="{html.escape(title)} parameter experiment comparison">
				<p class="caption">Each line is the median parameter-tuning convergence over the recorded seeds. The shaded band shows the interquartile range.</p>
				<div class="seed-grid">
					{''.join(seed_figure_blocks)}
				</div>
			</section>
			"""
		)

	(output_dir / "aggregate_summary.json").write_text(
		json.dumps(report_data, indent=2, sort_keys=True),
		encoding="utf-8",
	)
	(output_dir / "index.html").write_text(
		f"""<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Parameter-Tuning Experiment Comparison</title>
	<style>
		body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f8fa; }}
		header {{ padding: 28px 36px; background: #1f2933; color: white; }}
		main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
		section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
		h1, h2 {{ margin: 0 0 14px; }}
		p {{ margin: 4px 0 0; color: #bcccdc; }}
		p.caption {{ color: #52606d; margin-top: 8px; }}
		table {{ width: 100%; border-collapse: collapse; margin: 14px 0 18px; font-size: 14px; }}
		th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
		th:first-child, td:first-child {{ text-align: left; }}
		img {{ width: 100%; max-width: 960px; display: block; margin: 8px auto 12px; border: 1px solid #e5e7eb; }}
		.seed-grid {{ display: grid; grid-template-columns: repeat(1, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
		.seed-figure {{ border-top: 1px solid #e5e7eb; padding-top: 12px; }}
		h3 {{ margin: 0 0 8px; font-size: 15px; }}
	</style>
</head>
<body>
	<header>
		<h1>Parameter-Tuning Convergence Comparison</h1>
		<p>Three experiment lines per case. X-axis is unique valid configurations. Y-axis is best runtime so far, log scale.</p>
	</header>
	<main>
		{''.join(sections)}
	</main>
</body>
</html>
""",
		encoding="utf-8",
	)


def main() -> int:
	parser = argparse.ArgumentParser(description="Compare parameter-tuning convergence across experiment roots.")
	parser.add_argument(
		"--experiment",
		dest="experiments",
		action="append",
		required=True,
		help="Label and run root in the form label=path",
	)
	parser.add_argument("--output-dir", required=True)
	args = parser.parse_args()

	experiments: list[tuple[str, Path]] = []
	for item in args.experiments:
		if "=" not in item:
			raise SystemExit(f"invalid --experiment value: {item!r}")
		label, path = item.split("=", 1)
		experiments.append((label, Path(path)))
	build_report(experiments, Path(args.output_dir))
	print(Path(args.output_dir) / "index.html")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
