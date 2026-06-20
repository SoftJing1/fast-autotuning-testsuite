from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


CASES = (
	("gaussian_512", "Gaussian 512x512"),
	("gaussian_1024", "Gaussian 1024x1024"),
	("gaussian_2048", "Gaussian 2048x2048"),
	("gaussian_512x1024", "Gaussian 512x1024"),
	("gaussian_1024x512", "Gaussian 1024x512"),
	("gaussian_1024x2048", "Gaussian 1024x2048"),
	("gaussian_2048x1024", "Gaussian 2048x1024"),
	("gaussian_4096", "Gaussian 4096x4096"),
	("gaussian_2048x4096", "Gaussian 2048x4096"),
	("gaussian_4096x2048", "Gaussian 4096x2048"),
)


def _milestones(trace_path: Path, targets: tuple[int, ...] = (1, 10, 50, 100, 200, 500)) -> dict[int, tuple[int, float, str]]:
	rows = list(csv.DictReader(trace_path.open()))
	best = float("inf")
	best_exp = ""
	output = {}
	for row in rows:
		if row.get("runtime_ms") and row.get("exp_id"):
			runtime = float(row["runtime_ms"])
			if runtime < best:
				best = runtime
				best_exp = row["exp_id"]
		if not row.get("unique_config_count"):
			continue
		unique_count = int(row["unique_config_count"])
		for target in targets:
			if target not in output and unique_count >= target:
				output[target] = (int(row["iteration"]), best, best_exp)
	return output


def _trace_at_limit(trace_path: Path, valid_limit: int) -> dict:
	if not trace_path.exists():
		return {}
	rows = list(csv.DictReader(trace_path.open()))
	best = float("inf")
	best_exp = ""
	best_row = {}
	last_valid = {}
	for row in rows:
		if row.get("runtime_ms") and row.get("exp_id"):
			runtime = float(row["runtime_ms"])
			if runtime < best:
				best = runtime
				best_exp = row["exp_id"]
				best_row = dict(row)
		if row.get("unique_config_count"):
			last_valid = dict(row)
			if int(row["unique_config_count"]) >= valid_limit:
				break
	if not last_valid:
		return {}
	return {
		"unique_config_count": int(last_valid.get("unique_config_count") or 0),
		"valid_evaluation_count": int(last_valid.get("valid_evaluation_count") or 0),
		"iterations": int(last_valid.get("iteration") or 0),
		"best_runtime_ms": best,
		"best_exp_id": best_exp,
		"best_row": best_row,
	}


def _method_cell(summary: dict) -> str:
	if not summary:
		return "<td colspan='4'>missing</td>"
	return (
		f"<td>{summary.get('unique_config_count', '')}</td>"
		f"<td>{summary.get('valid_evaluation_count', '')}</td>"
		f"<td>{float(summary.get('best_runtime_ms', 0.0)):.6f}</td>"
		f"<td>{html.escape(str(summary.get('best_exp_id', '')))}</td>"
	)


def build_report(run_root: Path, output: Path, valid_limit: int) -> None:
	sections = []
	for case_dir_name, title in CASES:
		case_dir = run_root / case_dir_name
		parameter_trace = case_dir / "parameter_tuning" / "trace.csv"
		instruction_trace = case_dir / "instruction_map_tuning" / "trace.csv"
		parameter_summary = _trace_at_limit(parameter_trace, valid_limit)
		instruction_summary = _trace_at_limit(instruction_trace, valid_limit)
		plot_path = case_dir / f"convergence_unique_valid_{valid_limit}.png"
		plot_rel = plot_path.relative_to(output.parent)

		targets = tuple(value for value in (1, 10, 50, 100, 200, valid_limit) if value <= valid_limit)
		parameter_milestones = _milestones(parameter_trace, targets)
		instruction_milestones = _milestones(instruction_trace, targets)
		milestone_rows = []
		for target in targets:
			param = parameter_milestones.get(target)
			inst = instruction_milestones.get(target)
			if param is None or inst is None:
				continue
			milestone_rows.append(
				"<tr>"
				f"<td>{target}</td>"
				f"<td>{param[1]:.6f}</td><td>{html.escape(param[2])}</td><td>{param[0]}</td>"
				f"<td>{inst[1]:.6f}</td><td>{html.escape(inst[2])}</td><td>{inst[0]}</td>"
				"</tr>"
			)

		sections.append(
			f"""
			<section>
				<h2>{html.escape(title)}</h2>
				<table>
					<thead>
						<tr><th>Method</th><th>Unique valid</th><th>Evaluations</th><th>Best runtime (ms)</th><th>Best exp id</th></tr>
					</thead>
					<tbody>
						<tr><td>Parameter tuning</td>{_method_cell(parameter_summary)}</tr>
						<tr><td>Instruction-map tuning</td>{_method_cell(instruction_summary)}</tr>
					</tbody>
				</table>
				<img src="{html.escape(str(plot_rel))}" alt="{html.escape(title)} convergence plot">
				<table>
					<thead>
						<tr>
							<th>Unique valid</th>
							<th>Parameter best</th><th>Parameter exp</th><th>Parameter iter</th>
							<th>Instruction best</th><th>Instruction exp</th><th>Instruction iter</th>
						</tr>
					</thead>
					<tbody>
						{''.join(milestone_rows)}
					</tbody>
				</table>
			</section>
			"""
		)

	output.write_text(
		f"""<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Two-Step Autotuning {valid_limit}-Config Sweep</title>
	<style>
		body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f6f8fa; }}
		header {{ padding: 28px 36px; background: #1f2933; color: white; }}
		main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
		section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
		h1, h2 {{ margin: 0 0 14px; }}
		p {{ margin: 4px 0 0; color: #bcccdc; }}
		table {{ width: 100%; border-collapse: collapse; margin: 14px 0 18px; font-size: 14px; }}
		th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; }}
		th:first-child, td:first-child {{ text-align: left; }}
		img {{ width: 100%; max-width: 960px; display: block; margin: 8px auto 18px; border: 1px solid #e5e7eb; }}
	</style>
</head>
<body>
	<header>
		<h1>Two-Step Autotuning {valid_limit}-Config Sweep</h1>
		<p>Best runtime so far over unique valid dataset configurations.</p>
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
	parser = argparse.ArgumentParser(description="Generate a static HTML report for autotuning sweep results.")
	parser.add_argument("run_root")
	parser.add_argument("--output", default=None)
	parser.add_argument("--valid-limit", type=int, default=500)
	args = parser.parse_args()
	run_root = Path(args.run_root)
	output = Path(args.output) if args.output else run_root / "index.html"
	build_report(run_root, output, valid_limit=args.valid_limit)
	print(output)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
