from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path

from ..core.result_recorder import TRACE_FIELDS
from .aggregate_replicate_report import build_report


def _reduced_runtime(measured_samples: list[dict], statistic: str) -> float | None:
	values = [
		float(sample["runtime_ms"])
		for sample in measured_samples
		if sample.get("success") and sample.get("runtime_ms") is not None
	]
	if not values:
		return None
	if statistic == "median":
		return statistics.median(values)
	if statistic == "mean":
		return statistics.fmean(values)
	raise ValueError(f"unsupported runtime statistic: {statistic}")


def _load_runtime_map(runtime_samples_path: Path, statistic: str) -> dict[int, float]:
	runtime_by_exp_id: dict[int, float] = {}
	with runtime_samples_path.open() as f:
		for line in f:
			record = json.loads(line)
			exp_id = int(record["exp_id"])
			reduced = _reduced_runtime(record.get("measured_samples", []), statistic)
			if reduced is not None:
				runtime_by_exp_id[exp_id] = reduced
	return runtime_by_exp_id


def _coerce_value(value: str):
	if value == "":
		return None
	for caster in (int, float):
		try:
			return caster(value)
		except ValueError:
			continue
	return value


def _rewrite_method_dir(src_dir: Path, dst_dir: Path, statistic: str) -> None:
	dst_dir.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src_dir / "runtime_samples.jsonl", dst_dir / "runtime_samples.jsonl")
	if (src_dir / "final_config.json").exists():
		shutil.copy2(src_dir / "final_config.json", dst_dir / "final_config.json")
	if (src_dir / "live_experiments.db").exists():
		shutil.copy2(src_dir / "live_experiments.db", dst_dir / "live_experiments.db")

	run_config = json.loads((src_dir / "run_config.json").read_text())
	run_config["runtime_statistic"] = statistic
	run_config["derived_from"] = str(src_dir)
	(dst_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

	runtime_by_exp_id = _load_runtime_map(src_dir / "runtime_samples.jsonl", statistic)
	rows = list(csv.DictReader((src_dir / "trace.csv").open()))

	best_runtime: float | None = None
	best_row: dict | None = None
	out_rows: list[dict[str, str]] = []

	for row in rows:
		out = {field: row.get(field, "") for field in TRACE_FIELDS}
		exp_id_text = row.get("exp_id", "")
		runtime_text = row.get("runtime_ms", "")
		if exp_id_text:
			exp_id = int(exp_id_text)
			if exp_id in runtime_by_exp_id:
				runtime_text = str(runtime_by_exp_id[exp_id])
		out["runtime_ms"] = runtime_text
		if runtime_text:
			runtime = float(runtime_text)
			if best_runtime is None or runtime < best_runtime:
				best_runtime = runtime
				best_row = {
					key: _coerce_value(out[key])
					for key in (
						"candidate_status",
						"duplicate_count",
						"exp_id",
						"param_hash",
						"resolver_distance",
						"resolver_time_ms",
						"runtime_ms",
					)
					if key in out and out[key] != ""
				}
		out["best_runtime_ms"] = "" if best_runtime is None else str(best_runtime)
		out_rows.append(out)

	with (dst_dir / "trace.csv").open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
		writer.writeheader()
		writer.writerows(out_rows)

	source_summary = json.loads((src_dir / "summary.json").read_text())
	source_summary["best_runtime_ms"] = best_runtime
	source_summary["best_row"] = best_row
	source_summary["trace_path"] = str(dst_dir / "trace.csv")
	(dst_dir / "summary.json").write_text(json.dumps(source_summary, indent=2, sort_keys=True))


def derive_run_tree(src_root: Path, dst_root: Path, statistic: str) -> None:
	dst_root.mkdir(parents=True, exist_ok=True)
	if (src_root / "launcher.log").exists():
		shutil.copy2(src_root / "launcher.log", dst_root / "launcher.log")
	if (src_root / "run_replicates.sh").exists():
		shutil.copy2(src_root / "run_replicates.sh", dst_root / "run_replicates.sh")

	for case_dir in sorted(path for path in src_root.iterdir() if path.is_dir() and path.name != "logs"):
		for seed_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
			for method_name in ("parameter_tuning", "instruction_map_tuning"):
				src_method_dir = seed_dir / method_name
				if not (src_method_dir / "trace.csv").exists():
					continue
				dst_method_dir = dst_root / case_dir.name / seed_dir.name / method_name
				_rewrite_method_dir(src_method_dir, dst_method_dir, statistic)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Derive a new report tree by recomputing runtime from recorded raw samples."
	)
	parser.add_argument("src_root", help="Existing completed run root")
	parser.add_argument("dst_root", help="Output root for derived traces, summaries, and report")
	parser.add_argument(
		"--statistic",
		choices=["median", "mean"],
		default="median",
		help="Runtime reducer applied to measured samples from runtime_samples.jsonl.",
	)
	args = parser.parse_args()

	src_root = Path(args.src_root)
	dst_root = Path(args.dst_root)
	derive_run_tree(src_root, dst_root, args.statistic)
	build_report(dst_root, dst_root / "index_aggregate.html")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
