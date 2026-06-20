from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .aggregate_replicate_report import CASES, _format_float
from .plot_comparison import _dedupe_x, _load_series


MILESTONES = (1, 10, 50, 100, 200, 300)


@dataclass(frozen=True)
class RunResult:
	best_runtime_ms: float
	valid_evaluation_count: int
	unique_config_count: int
	iterations: int
	wall_time_s: float
	best_status: str
	best_exp_id: str
	trace_path: Path


@dataclass(frozen=True)
class PairResult:
	case_name: str
	case_title: str
	seed: int
	dbonly: RunResult | None
	mutation: RunResult | None

	@property
	def complete(self) -> bool:
		return self.dbonly is not None and self.mutation is not None

	@property
	def improvement_pct(self) -> float:
		if not self.complete or self.dbonly is None or self.mutation is None:
			return math.nan
		if self.dbonly.best_runtime_ms <= 0:
			return math.nan
		return (self.dbonly.best_runtime_ms - self.mutation.best_runtime_ms) / self.dbonly.best_runtime_ms * 100.0

	@property
	def winner(self) -> str:
		if not self.complete or self.dbonly is None or self.mutation is None:
			return "missing"
		if math.isclose(self.dbonly.best_runtime_ms, self.mutation.best_runtime_ms, rel_tol=1e-6, abs_tol=1e-9):
			return "tie"
		return "mutation64" if self.mutation.best_runtime_ms < self.dbonly.best_runtime_ms else "db-only"


def _load_result(method_dir: Path) -> RunResult | None:
	summary_path = method_dir / "summary.json"
	trace_path = method_dir / "trace.csv"
	if not summary_path.exists() or not trace_path.exists():
		return None
	summary = json.loads(summary_path.read_text())
	best_row = summary.get("best_row") or {}
	return RunResult(
		best_runtime_ms=float(summary.get("best_runtime_ms", math.nan)),
		valid_evaluation_count=int(summary.get("valid_evaluation_count", 0)),
		unique_config_count=int(summary.get("unique_config_count", 0)),
		iterations=int(summary.get("iterations", 0)),
		wall_time_s=float(summary.get("wall_time_s", math.nan)),
		best_status=str(best_row.get("candidate_status", "")),
		best_exp_id=str(best_row.get("exp_id", "")),
		trace_path=trace_path,
	)


def _series_at_milestones(trace_path: Path) -> dict[int, float]:
	values: dict[int, float] = {}
	if not trace_path.exists():
		return values
	with trace_path.open() as f:
		for row in csv.DictReader(f):
			x_text = row.get("valid_evaluation_count", "")
			y_text = row.get("best_runtime_ms", "")
			if not x_text or not y_text:
				continue
			x_value = int(x_text)
			for milestone in MILESTONES:
				if milestone not in values and x_value >= milestone:
					values[milestone] = float(y_text)
	return values


def _plot_pair(pair: PairResult, output_path: Path) -> None:
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:
		raise RuntimeError("matplotlib is required to plot comparison graphs") from exc

	fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8))
	colors = {"DB-only": "#315f72", "Mutation64": "#b6412d"}
	for label, result in (("DB-only", pair.dbonly), ("Mutation64", pair.mutation)):
		if result is None:
			continue
		x_values, y_values = _load_series(result.trace_path, "valid")
		x_values, y_values = _dedupe_x(x_values, y_values)
		ax.plot(x_values, y_values, label=label, linewidth=2.2, color=colors[label])
	ax.set_title(f"{pair.case_title} - Seed {pair.seed}")
	ax.set_xlabel("Valid evaluation")
	ax.set_ylabel("Best runtime so far (ms)")
	ax.set_yscale("log")
	ax.grid(True, alpha=0.25)
	ax.legend()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def _cell_class(pair: PairResult) -> str:
	value = pair.improvement_pct
	if not math.isfinite(value):
		return "missing"
	if value > 1.0:
		return "win"
	if value < -1.0:
		return "loss"
	return "tie"


def _format_pct(value: float) -> str:
	if not math.isfinite(value):
		return "n/a"
	return f"{value:+.2f}%"


def _format_seconds(value: float) -> str:
	if not math.isfinite(value):
		return "n/a"
	return f"{value / 60.0:.1f} min"


def _collect_pairs(dbonly_root: Path, mutation_root: Path) -> list[PairResult]:
	pairs: list[PairResult] = []
	for case_name, case_title in CASES:
		for seed in range(1, 6):
			pairs.append(
				PairResult(
					case_name=case_name,
					case_title=case_title,
					seed=seed,
					dbonly=_load_result(dbonly_root / case_name / f"seed_{seed}" / "instruction_map_tuning"),
					mutation=_load_result(mutation_root / case_name / f"seed_{seed}" / "instruction_map_tuning"),
				)
			)
	return pairs


def _summary_stats(pairs: list[PairResult]) -> dict[str, object]:
	complete = [pair for pair in pairs if pair.complete]
	improvements = [pair.improvement_pct for pair in complete if math.isfinite(pair.improvement_pct)]
	return {
		"paired_runs": len(complete),
		"mutation_wins": sum(1 for pair in complete if pair.winner == "mutation64"),
		"dbonly_wins": sum(1 for pair in complete if pair.winner == "db-only"),
		"ties": sum(1 for pair in complete if pair.winner == "tie"),
		"median_improvement_pct": median(improvements) if improvements else math.nan,
		"mean_improvement_pct": sum(improvements) / len(improvements) if improvements else math.nan,
	}


def _matrix_html(pairs: list[PairResult]) -> str:
	by_key = {(pair.case_name, pair.seed): pair for pair in pairs}
	rows: list[str] = []
	for case_name, case_title in CASES:
		cells = []
		case_pairs = [by_key[(case_name, seed)] for seed in range(1, 6)]
		improvements = [pair.improvement_pct for pair in case_pairs if math.isfinite(pair.improvement_pct)]
		case_median = median(improvements) if improvements else math.nan
		for seed in range(1, 6):
			pair = by_key[(case_name, seed)]
			cells.append(
				f"""<td class="{_cell_class(pair)}">
					<a href="#{html.escape(case_name)}-seed-{seed}">
						<span>{html.escape(pair.winner)}</span>
						<strong>{html.escape(_format_pct(pair.improvement_pct))}</strong>
					</a>
				</td>"""
			)
		rows.append(
			"<tr>"
			f"<th>{html.escape(case_title)}</th>"
			f"{''.join(cells)}"
			f"<td>{html.escape(_format_pct(case_median))}</td>"
			"</tr>"
		)
	return "".join(rows)


def _detail_html(pair: PairResult, plot_rel: Path) -> str:
	if pair.dbonly is None or pair.mutation is None:
		return f"<section id=\"{html.escape(pair.case_name)}-seed-{pair.seed}\"><h2>{html.escape(pair.case_title)} Seed {pair.seed}</h2><p>Missing paired data.</p></section>"

	db_milestones = _series_at_milestones(pair.dbonly.trace_path)
	mut_milestones = _series_at_milestones(pair.mutation.trace_path)
	milestone_rows = []
	for milestone in MILESTONES:
		db_value = db_milestones.get(milestone, math.nan)
		mut_value = mut_milestones.get(milestone, math.nan)
		delta = math.nan
		if math.isfinite(db_value) and db_value > 0 and math.isfinite(mut_value):
			delta = (db_value - mut_value) / db_value * 100.0
		milestone_rows.append(
			"<tr>"
			f"<td>{milestone}</td>"
			f"<td>{_format_float(db_value)}</td>"
			f"<td>{_format_float(mut_value)}</td>"
			f"<td>{html.escape(_format_pct(delta))}</td>"
			"</tr>"
		)
	best_rows = (
		"<tr>"
		"<td>DB-only</td>"
		f"<td>{_format_float(pair.dbonly.best_runtime_ms)}</td>"
		f"<td>{pair.dbonly.valid_evaluation_count}</td>"
		f"<td>{pair.dbonly.unique_config_count}</td>"
		f"<td>{html.escape(pair.dbonly.best_status)}</td>"
		f"<td>{html.escape(pair.dbonly.best_exp_id)}</td>"
		f"<td>{html.escape(_format_seconds(pair.dbonly.wall_time_s))}</td>"
		"</tr>"
		"<tr>"
		"<td>Mutation64</td>"
		f"<td>{_format_float(pair.mutation.best_runtime_ms)}</td>"
		f"<td>{pair.mutation.valid_evaluation_count}</td>"
		f"<td>{pair.mutation.unique_config_count}</td>"
		f"<td>{html.escape(pair.mutation.best_status)}</td>"
		f"<td>{html.escape(pair.mutation.best_exp_id)}</td>"
		f"<td>{html.escape(_format_seconds(pair.mutation.wall_time_s))}</td>"
		"</tr>"
	)
	return f"""
	<section id="{html.escape(pair.case_name)}-seed-{pair.seed}">
		<div class="section-head">
			<h2>{html.escape(pair.case_title)} Seed {pair.seed}</h2>
			<a href="#matrix">Back to matrix</a>
		</div>
		<div class="verdict {_cell_class(pair)}">
			<span>{html.escape(pair.winner)}</span>
			<strong>{html.escape(_format_pct(pair.improvement_pct))}</strong>
		</div>
		<img src="{html.escape(str(plot_rel))}" alt="{html.escape(pair.case_title)} seed {pair.seed} DB-only vs mutation64 convergence">
		<table>
			<thead>
				<tr><th>Experiment</th><th>Final best ms</th><th>Valid evals</th><th>Unique configs</th><th>Best status</th><th>Best exp</th><th>Wall time</th></tr>
			</thead>
			<tbody>{best_rows}</tbody>
		</table>
		<table>
			<thead>
				<tr><th>Valid eval</th><th>DB-only best ms</th><th>Mutation64 best ms</th><th>Mutation delta</th></tr>
			</thead>
			<tbody>{''.join(milestone_rows)}</tbody>
		</table>
	</section>
	"""


def build_report(dbonly_root: Path, mutation_root: Path, output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	pairs = _collect_pairs(dbonly_root, mutation_root)
	stats = _summary_stats(pairs)
	detail_sections: list[str] = []
	for pair in pairs:
		plot_path = output_dir / "plots" / pair.case_name / f"seed_{pair.seed}.png"
		if pair.complete:
			_plot_pair(pair, plot_path)
		detail_sections.append(_detail_html(pair, plot_path.relative_to(output_dir)))

	payload = {
		"dbonly_root": str(dbonly_root),
		"mutation_root": str(mutation_root),
		"summary": stats,
		"pairs": [
			{
				"case": pair.case_name,
				"seed": pair.seed,
				"winner": pair.winner,
				"improvement_pct": pair.improvement_pct,
				"dbonly_best_runtime_ms": None if pair.dbonly is None else pair.dbonly.best_runtime_ms,
				"mutation_best_runtime_ms": None if pair.mutation is None else pair.mutation.best_runtime_ms,
			}
			for pair in pairs
		],
	}
	(output_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
	index_path = output_dir / "index.html"
	index_path.write_text(
		f"""<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Gaussian DB-only vs Mutation64</title>
	<style>
		body {{ margin: 0; font-family: Arial, sans-serif; color: #172026; background: #f5f7f8; }}
		header {{ padding: 26px 34px; background: #20343d; color: white; }}
		main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
		h1, h2 {{ margin: 0 0 10px; }}
		p {{ margin: 4px 0; }}
		section {{ background: white; border: 1px solid #d7dee2; border-radius: 6px; padding: 18px; margin-bottom: 20px; }}
		.summary {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
		.metric {{ background: #edf2f4; border-radius: 6px; padding: 12px; }}
		.metric span {{ display: block; color: #52646d; font-size: 12px; }}
		.metric strong {{ display: block; margin-top: 5px; font-size: 20px; }}
		table {{ width: 100%; border-collapse: collapse; margin: 12px 0 16px; font-size: 14px; }}
		th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8eb; text-align: right; vertical-align: middle; }}
		th:first-child, td:first-child {{ text-align: left; }}
		.matrix td {{ width: 14%; padding: 0; }}
		.matrix a {{ display: block; min-height: 52px; padding: 8px 10px; color: inherit; text-decoration: none; }}
		.matrix span, .verdict span {{ display: block; font-size: 12px; color: #52646d; }}
		.matrix strong, .verdict strong {{ display: block; margin-top: 5px; font-size: 15px; }}
		.win {{ background: #dff3e3; }}
		.loss {{ background: #fde3dc; }}
		.tie {{ background: #eef1f3; }}
		.missing {{ background: #f8f1d9; }}
		img {{ width: 100%; max-width: 940px; display: block; margin: 12px auto 18px; border: 1px solid #dfe5e8; }}
		.section-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }}
		.section-head a {{ color: #315f72; }}
		.verdict {{ display: inline-block; border-radius: 6px; padding: 8px 12px; min-width: 118px; margin-bottom: 8px; }}
		@media (max-width: 860px) {{
			.summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
			main {{ padding: 12px; }}
			table {{ font-size: 12px; }}
			th, td {{ padding: 7px 6px; }}
		}}
	</style>
</head>
<body>
	<header>
		<h1>Gaussian DB-only vs Mutation64</h1>
		<p>Positive delta means mutation64 found a lower final best runtime than the DB-only instruction-map run.</p>
	</header>
	<main>
		<section>
			<h2>Summary</h2>
			<div class="summary">
				<div class="metric"><span>Paired runs</span><strong>{stats["paired_runs"]}/50</strong></div>
				<div class="metric"><span>Mutation wins</span><strong>{stats["mutation_wins"]}</strong></div>
				<div class="metric"><span>DB-only wins</span><strong>{stats["dbonly_wins"]}</strong></div>
				<div class="metric"><span>Ties</span><strong>{stats["ties"]}</strong></div>
				<div class="metric"><span>Median mutation delta</span><strong>{html.escape(_format_pct(float(stats["median_improvement_pct"])))}</strong></div>
			</div>
		</section>
		<section id="matrix">
			<h2>Matrix</h2>
			<table class="matrix">
				<thead>
					<tr><th>Case</th><th>Seed 1</th><th>Seed 2</th><th>Seed 3</th><th>Seed 4</th><th>Seed 5</th><th>Median</th></tr>
				</thead>
				<tbody>{_matrix_html(pairs)}</tbody>
			</table>
		</section>
		{''.join(detail_sections)}
	</main>
</body>
</html>
""",
		encoding="utf-8",
	)
	return index_path


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate a mutation impact report comparing paired instruction-map runs.")
	parser.add_argument("--dbonly-root", required=True)
	parser.add_argument("--mutation-root", required=True)
	parser.add_argument("--output-dir", required=True)
	args = parser.parse_args()
	index_path = build_report(Path(args.dbonly_root), Path(args.mutation_root), Path(args.output_dir))
	print(index_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
