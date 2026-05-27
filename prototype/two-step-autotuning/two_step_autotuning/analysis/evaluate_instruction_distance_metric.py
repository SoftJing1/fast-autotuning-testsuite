from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from statistics import mean, median

from ..core.dataset import TuningDataset
from ..core.instruction_distance import InstructionDistanceModel, instruction_neighbor_results_to_dicts
DEFAULT_DB = os.environ.get(
	"TWO_STEP_AUTOTUNING_DB",
	"experiments/exp_20260415_large_scale_5000cfg/experiments.db",
)


def _parse_case(value: str) -> tuple[str, str]:
	if ":" not in value:
		raise argparse.ArgumentTypeError("case must be formatted as kernel:input_size")
	kernel, input_size = value.split(":", 1)
	if not kernel or not input_size:
		raise argparse.ArgumentTypeError("case must be formatted as kernel:input_size")
	return kernel, input_size


def _percentile(values: list[float], percentile: float) -> float:
	if not values:
		return 0.0
	ordered = sorted(values)
	index = round((len(ordered) - 1) * percentile)
	return float(ordered[index])


def _summarize(case: tuple[str, str], rows: list[dict]) -> dict:
	abs_diffs = [float(row["abs_runtime_diff_ms"]) for row in rows]
	relative_diffs = [float(row["relative_runtime_diff"]) for row in rows]
	tie_counts = [int(row["tie_count"]) for row in rows]
	zero_distance_count = sum(1 for row in rows if row["zero_distance_neighbor"])
	return {
		"kernel": case[0],
		"input_size": case[1],
		"metric": "instruction_map",
		"base_count": len(rows),
		"mean_abs_runtime_diff_ms": mean(abs_diffs),
		"median_abs_runtime_diff_ms": median(abs_diffs),
		"p90_abs_runtime_diff_ms": _percentile(abs_diffs, 0.90),
		"mean_relative_runtime_diff_pct": 100.0 * mean(relative_diffs),
		"median_relative_runtime_diff_pct": 100.0 * median(relative_diffs),
		"p90_relative_runtime_diff_pct": 100.0 * _percentile(relative_diffs, 0.90),
		"mean_tie_count": mean(tie_counts),
		"max_tie_count": max(tie_counts),
		"zero_distance_neighbor_count": zero_distance_count,
		"zero_distance_neighbor_pct": 100.0 * zero_distance_count / len(rows),
	}


def _write_csv(path: Path, rows: list[dict]) -> None:
	if not rows:
		return
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def evaluate_case(
	db_path: Path,
	case: tuple[str, str],
	sample_count: int | None,
	random_seed: int,
) -> tuple[list[dict], dict]:
	dataset = TuningDataset(db_path, case[0], case[1])
	model = InstructionDistanceModel(dataset)
	base_indices = list(range(len(dataset.records)))
	if sample_count is not None and sample_count < len(base_indices):
		rng = random.Random(random_seed)
		base_indices = sorted(rng.sample(base_indices, sample_count))

	results = model.nearest_neighbors(base_indices)
	rows = instruction_neighbor_results_to_dicts(results)
	for row in rows:
		row["kernel"] = case[0]
		row["input_size"] = case[1]
	summary = _summarize(case, rows)
	return rows, summary


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Evaluate instruction-map distance by nearest-neighbor runtime similarity."
	)
	parser.add_argument("--db", default=DEFAULT_DB)
	parser.add_argument(
		"--case",
		type=_parse_case,
		action="append",
		default=[],
		help="Kernel/input pair, for example gaussian:512x512. May be repeated.",
	)
	parser.add_argument("--sample-count", type=int, default=None)
	parser.add_argument("--random-seed", type=int, default=1)
	parser.add_argument(
		"--output-dir",
		default="prototype/two-step-autotuning/runs/instruction_distance_metric",
	)
	return parser


def main() -> int:
	parser = build_argparser()
	args = parser.parse_args()
	db_path = Path(args.db)
	cases = args.case or [("gaussian", "512x512"), ("gemm", "128x128x128")]
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	all_details = []
	all_summaries = []
	for case in cases:
		details, summary = evaluate_case(
			db_path=db_path,
			case=case,
			sample_count=args.sample_count,
			random_seed=args.random_seed,
		)
		all_details.extend(details)
		all_summaries.append(summary)

	_write_csv(output_dir / "nearest_neighbors.csv", all_details)
	_write_csv(output_dir / "summary.csv", all_summaries)
	(output_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2, sort_keys=True))

	for row in all_summaries:
		print(
			f"{row['kernel']} {row['input_size']} instruction_map: "
			f"median rel diff {row['median_relative_runtime_diff_pct']:.2f}%, "
			f"mean rel diff {row['mean_relative_runtime_diff_pct']:.2f}%, "
			f"median abs diff {row['median_abs_runtime_diff_ms']:.4f} ms, "
			f"zero-distance neighbors {row['zero_distance_neighbor_pct']:.1f}%"
		)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
