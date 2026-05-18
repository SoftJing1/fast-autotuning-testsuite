from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


TRACE_FIELDS = [
	"iteration",
	"wall_time_s",
	"candidate_status",
	"runtime_ms",
	"best_runtime_ms",
	"exp_id",
	"param_hash",
	"resolver_distance",
	"raw_distance",
	"mix_distance",
	"total_distance",
	"resolver_time_ms",
	"duplicate_count",
	"valid_evaluation_count",
	"unique_config_count",
]


class ResultRecorder:
	def __init__(self, output_dir: str | Path, metadata: dict[str, Any]):
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.trace_path = self.output_dir / "trace.csv"
		self.summary_path = self.output_dir / "summary.json"
		self.start = time.perf_counter()
		self.iteration = 0
		self.best_runtime_ms: float | None = None
		self.best_row: dict[str, Any] | None = None
		self.seen_exp_ids: set[int] = set()
		self.valid_evaluation_count = 0
		(self.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
		with self.trace_path.open("w", newline="") as f:
			csv.DictWriter(f, fieldnames=TRACE_FIELDS).writeheader()

	def record(self, row: dict[str, Any]) -> None:
		self.iteration += 1
		runtime = row.get("runtime_ms")
		exp_id = row.get("exp_id")
		if exp_id not in (None, ""):
			self.valid_evaluation_count += 1
			self.seen_exp_ids.add(int(exp_id))
		if runtime is not None:
			runtime = float(runtime)
			if self.best_runtime_ms is None or runtime < self.best_runtime_ms:
				self.best_runtime_ms = runtime
				self.best_row = dict(row)
		out = {field: "" for field in TRACE_FIELDS}
		out.update(row)
		out["iteration"] = self.iteration
		out["wall_time_s"] = time.perf_counter() - self.start
		out["best_runtime_ms"] = "" if self.best_runtime_ms is None else self.best_runtime_ms
		out["valid_evaluation_count"] = self.valid_evaluation_count
		out["unique_config_count"] = len(self.seen_exp_ids)
		with self.trace_path.open("a", newline="") as f:
			csv.DictWriter(f, fieldnames=TRACE_FIELDS).writerow(out)

	def write_summary(self) -> None:
		payload = {
			"iterations": self.iteration,
			"wall_time_s": time.perf_counter() - self.start,
			"best_runtime_ms": self.best_runtime_ms,
			"unique_config_count": len(self.seen_exp_ids),
			"valid_evaluation_count": self.valid_evaluation_count,
			"best_row": self.best_row,
			"trace_path": str(self.trace_path),
		}
		self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
