from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelRecord:
	exp_id: int
	kernel_type: str
	input_size: str
	param_hash: str
	config: dict[str, Any]
	runtime_ms: float
	raw_counts: dict[str, int]

	@property
	def total_count(self) -> int:
		return sum(int(value) for value in self.raw_counts.values())


@dataclass(frozen=True)
class LiveProfile:
	exp_id: int
	kernel_type: str
	input_size: str
	param_hash: str
	config: dict[str, Any]
	raw_counts: dict[str, int]
	runtime_ms: float | None = None

	@property
	def total_count(self) -> int:
		return sum(int(value) for value in self.raw_counts.values())


@dataclass(frozen=True)
class ResolutionResult:
	status: str
	record: KernelRecord | None
	distance: float | None = None
	raw_distance: float | None = None
	mix_distance: float | None = None
	total_distance: float | None = None
	duplicate_count: int = 0
	resolver_time_ms: float = 0.0

	@property
	def valid(self) -> bool:
		return self.record is not None and self.status in {"valid", "ambiguous"}
