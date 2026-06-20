from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from statistics import median
from pathlib import Path
from typing import Any

from .instruction_map import canonical_config_key, sum_counts
from .types import KernelRecord


INPUT_KEYS_BY_KERNEL = {
	"gemm": {"M", "N", "K", "input_size_l_1", "input_size_l_2", "input_size_r_1"},
	"gaussian": {"input_size_h", "input_size_w", "input_size_1", "input_size_2"},
}


class TuningDataset:
	def __init__(
		self,
		db_path: str | Path,
		kernel_type: str,
		input_size: str,
		require_runtime: bool = True,
	):
		self.db_path = Path(db_path)
		self.kernel_type = kernel_type
		self.input_size = input_size
		self.require_runtime = require_runtime
		self.records = self._load_records()
		if not self.records:
			status_label = "completed" if self.require_runtime else "completed/profiled"
			raise ValueError(
				f"No {status_label} records with instruction counts for {kernel_type}/{input_size} in {self.db_path}"
			)
		self.opcodes = sorted({op for record in self.records for op in record.raw_counts})
		self.config_by_key = {canonical_config_key(record.config): record for record in self.records}
		self.records_by_instruction_key = self._group_by_instruction_key()
		runtime_records = [record for record in self.records if record.runtime_ms is not None]
		self.best_record = (
			min(runtime_records, key=lambda record: float(record.runtime_ms))
			if runtime_records
			else self.records[0]
		)

	def _load_records(self) -> list[KernelRecord]:
		conn = sqlite3.connect(str(self.db_path))
		conn.row_factory = sqlite3.Row
		try:
			runtime_filter = "AND e.runtime_ms IS NOT NULL" if self.require_runtime else ""
			status_filter = "e.status = 'completed'" if self.require_runtime else "e.status IN ('completed', 'profiled')"
			rows = conn.execute(
				"""
				SELECT
					e.exp_id,
					e.kernel_type,
					e.input_size,
					e.param_hash,
					e.config_json,
					e.runtime_ms,
					lic.template_name,
					lic.total_instruction_counts_json
				FROM experiments e
				JOIN llvm_instruction_counts lic ON lic.exp_id = e.exp_id
				WHERE {status_filter}
				  {runtime_filter}
				  AND e.kernel_type = ?
				  AND e.input_size = ?
				ORDER BY e.exp_id ASC, lic.template_name ASC
				""".format(status_filter=status_filter, runtime_filter=runtime_filter),
				(self.kernel_type, self.input_size),
			).fetchall()
		finally:
			conn.close()

		grouped: dict[int, dict[str, Any]] = {}
		counts_by_exp: dict[int, list[dict[str, int]]] = defaultdict(list)
		for row in rows:
			exp_id = int(row["exp_id"])
			grouped.setdefault(
				exp_id,
				{
					"exp_id": exp_id,
					"kernel_type": row["kernel_type"],
					"input_size": row["input_size"],
					"param_hash": row["param_hash"],
					"config": json.loads(row["config_json"]),
					"runtime_ms": None if row["runtime_ms"] is None else float(row["runtime_ms"]),
				},
			)
			counts_by_exp[exp_id].append(json.loads(row["total_instruction_counts_json"]))

		records = []
		for exp_id, item in grouped.items():
			count_maps = counts_by_exp[exp_id]
			if self.kernel_type == "gemm" and len(count_maps) < 2:
				continue
			records.append(
				KernelRecord(
					exp_id=item["exp_id"],
					kernel_type=item["kernel_type"],
					input_size=item["input_size"],
					param_hash=item["param_hash"],
					config=item["config"],
					runtime_ms=item["runtime_ms"],
					raw_counts=sum_counts(count_maps),
				)
			)
		return records

	def _group_by_instruction_key(self) -> dict[str, list[KernelRecord]]:
		groups: dict[str, list[KernelRecord]] = defaultdict(list)
		for record in self.records:
			groups[self.instruction_key(record.raw_counts)].append(record)
		return dict(groups)

	def instruction_key(self, counts: dict[str, Any]) -> str:
		return "|".join(f"{op}={int(counts.get(op, 0))}" for op in self.opcodes)

	def lookup_config(self, config: dict[str, Any]) -> KernelRecord | None:
		return self.config_by_key.get(canonical_config_key(config))

	def tuning_parameter_names(self) -> list[str]:
		input_keys = INPUT_KEYS_BY_KERNEL.get(self.kernel_type, set())
		keys = set()
		for record in self.records:
			keys.update(record.config)
		return sorted(key for key in keys if key not in input_keys)

	def fixed_input_config(self) -> dict[str, Any]:
		input_keys = INPUT_KEYS_BY_KERNEL.get(self.kernel_type, set())
		for record in self.records:
			return {key: value for key, value in record.config.items() if key in input_keys}
		return {}

	def observed_parameter_values(self) -> dict[str, list[Any]]:
		values: dict[str, set[Any]] = {name: set() for name in self.tuning_parameter_names()}
		for record in self.records:
			for name in values:
				values[name].add(record.config[name])
		return {name: sorted(items) for name, items in values.items()}

	def top_records(self, count: int) -> list[KernelRecord]:
		return sorted(
			self.records,
			key=lambda record: (
				record.runtime_ms is None,
				float("inf") if record.runtime_ms is None else float(record.runtime_ms),
				record.exp_id,
			),
		)[:count]

	def sampled_records(self, count: int, seed: int) -> list[KernelRecord]:
		if count <= 0:
			return []
		rng = random.Random(seed)
		if count >= len(self.records):
			records = list(self.records)
			rng.shuffle(records)
			return records
		return rng.sample(self.records, count)

	def median_instruction_counts(self) -> dict[str, int]:
		from .instruction_space import filter_instruction_opcodes

		return {
			op: int(median(int(record.raw_counts.get(op, 0)) for record in self.records))
			for op in filter_instruction_opcodes(self.opcodes)
		}
