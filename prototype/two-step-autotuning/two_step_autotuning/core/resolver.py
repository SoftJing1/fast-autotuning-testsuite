from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import TuningDataset
from .instruction_map import log1p
from .instruction_space import EXCLUDED_INSTRUCTION_OPCODES, filter_instruction_opcodes
from .types import KernelRecord, LiveProfile, ResolutionResult


@dataclass(frozen=True)
class ResolverWeights:
	raw: float = 1.0
	normalized: float = 0.0
	total: float = 0.0


@dataclass(frozen=True)
class _DistanceResult:
	index: int | None
	distance: float | None = None
	raw_distance: float | None = None
	mix_distance: float | None = None
	total_distance: float | None = None
	duplicate_count: int = 0
	resolver_time_ms: float = 0.0

	@property
	def valid(self) -> bool:
		return self.index is not None


class _InstructionDistanceIndex:
	def __init__(self, items: list[Any], opcodes: list[str], weights: ResolverWeights):
		self.weights = weights
		self.items: list[Any] = []
		self.opcodes = opcodes
		self._count_matrix: np.ndarray | None = None
		self.rebuild(items, opcodes)

	def rebuild(self, items: list[Any], opcodes: list[str]) -> None:
		self.items = list(items)
		self.opcodes = list(opcodes)
		if not self.items:
			self._count_matrix = None
			return
		self._count_matrix = np.array(
			[[max(float(item.raw_counts.get(op, 0)), 0.0) for op in self.opcodes] for item in self.items],
			dtype=float,
		)

	def nearest(self, requested_counts: dict[str, Any], max_distance: float | None = None) -> _DistanceResult:
		start = time.perf_counter()
		if not self.items or self._count_matrix is None:
			return _DistanceResult(index=None, resolver_time_ms=0.0)

		request_vector = np.array(
			[max(float(requested_counts.get(op, 0)), 0.0) for op in self.opcodes],
			dtype=float,
		)
		raw_dist = np.sqrt(((self._count_matrix - request_vector) ** 2).sum(axis=1))
		combined = raw_dist
		best_index = int(np.argmin(combined))
		best_distance = float(combined[best_index])
		if max_distance is not None and best_distance > max_distance:
			best = None
		else:
			best = best_index
		return _DistanceResult(
			index=best,
			distance=best_distance,
			raw_distance=float(raw_dist[best_index]),
			mix_distance=0.0,
			total_distance=0.0,
			duplicate_count=int(np.isclose(combined, best_distance, rtol=0.0, atol=1.0e-12).sum()),
			resolver_time_ms=(time.perf_counter() - start) * 1000.0,
		)

	def distance_to_counts(self, requested_counts: dict[str, Any], candidate_counts: dict[str, Any]) -> _DistanceResult:
		start = time.perf_counter()
		if self._count_matrix is None or not self.opcodes:
			return _DistanceResult(index=None, resolver_time_ms=0.0)

		request_vector = np.array(
			[max(float(requested_counts.get(op, 0)), 0.0) for op in self.opcodes],
			dtype=float,
		)
		candidate_vector = np.array(
			[max(float(candidate_counts.get(op, 0)), 0.0) for op in self.opcodes],
			dtype=float,
		)
		raw_distance = float(np.sqrt(((candidate_vector - request_vector) ** 2).sum()))
		return _DistanceResult(
			index=0,
			distance=raw_distance,
			raw_distance=raw_distance,
			mix_distance=0.0,
			total_distance=0.0,
			duplicate_count=1,
			resolver_time_ms=(time.perf_counter() - start) * 1000.0,
		)


class DatabaseApproxInstructionMapResolver:
	"""Approximate resolver constrained to existing dataset records."""

	def __init__(
		self,
		dataset: TuningDataset,
		weights: ResolverWeights | None = None,
		max_distance: float | None = None,
		excluded_opcodes: set[str] | frozenset[str] | None = None,
	):
		self.dataset = dataset
		self.weights = weights or ResolverWeights()
		self.max_distance = max_distance
		self.excluded_opcodes = (
			EXCLUDED_INSTRUCTION_OPCODES if excluded_opcodes is None else frozenset(excluded_opcodes)
		)
		self.opcodes = filter_instruction_opcodes(dataset.opcodes)
		self._records = dataset.records
		self._distance_index = _InstructionDistanceIndex(self._records, self.opcodes, self.weights)
		self._records_by_instruction_key = self._group_by_instruction_key()

	def _instruction_key(self, counts: dict[str, Any]) -> str:
		return "|".join(f"{op}={int(counts.get(op, 0))}" for op in self.opcodes)

	def _group_by_instruction_key(self) -> dict[str, list[KernelRecord]]:
		groups: dict[str, list[KernelRecord]] = {}
		for record in self._records:
			groups.setdefault(self._instruction_key(record.raw_counts), []).append(record)
		return groups

	def resolve(self, requested_counts: dict[str, Any]) -> ResolutionResult:
		match = self._distance_index.nearest(requested_counts, self.max_distance)
		if not match.valid:
			return ResolutionResult(
				status="invalid",
				record=None,
				distance=match.distance,
				raw_distance=match.raw_distance,
				mix_distance=match.mix_distance,
				total_distance=match.total_distance,
				resolver_time_ms=match.resolver_time_ms,
			)

		nearest_record = self._records[match.index]
		key = self._instruction_key(nearest_record.raw_counts)
		candidates = self._records_by_instruction_key.get(key, [nearest_record])
		status = "ambiguous" if len(candidates) > 1 else "valid"
		return ResolutionResult(
			status=status,
			record=nearest_record,
			distance=match.distance,
			raw_distance=match.raw_distance,
			mix_distance=match.mix_distance,
			total_distance=match.total_distance,
			duplicate_count=len(candidates),
			resolver_time_ms=match.resolver_time_ms,
		)


@dataclass(frozen=True)
class LiveResolutionResult:
	status: str
	profile: LiveProfile | None
	distance: float | None = None
	raw_distance: float | None = None
	mix_distance: float | None = None
	total_distance: float | None = None
	duplicate_count: int = 0
	resolver_time_ms: float = 0.0

	@property
	def valid(self) -> bool:
		return self.profile is not None and self.status in {"valid", "ambiguous"}


class OnlineInstructionMapResolver:
	def __init__(
		self,
		weights: ResolverWeights | None = None,
		max_distance: float | None = None,
	):
		self.weights = weights or ResolverWeights()
		self.max_distance = max_distance
		self.profiles: list[LiveProfile] = []
		self.opcodes: list[str] = []
		self._distance_index = _InstructionDistanceIndex([], [], self.weights)

	def add_profiles(self, profiles: list[LiveProfile]) -> None:
		existing = {profile.param_hash for profile in self.profiles}
		added = [profile for profile in profiles if profile.param_hash not in existing]
		if not added:
			return
		self.profiles.extend(added)
		self.opcodes = filter_instruction_opcodes(
			{op for profile in self.profiles for op in profile.raw_counts}
		)
		self._distance_index.rebuild(self.profiles, self.opcodes)

	def resolve(self, requested_counts: dict[str, Any]) -> LiveResolutionResult:
		match = self._distance_index.nearest(requested_counts, self.max_distance)
		if not match.valid:
			return LiveResolutionResult(
				"invalid",
				None,
				distance=match.distance,
				raw_distance=match.raw_distance,
				mix_distance=match.mix_distance,
				total_distance=match.total_distance,
				duplicate_count=match.duplicate_count,
				resolver_time_ms=match.resolver_time_ms,
			)

		return LiveResolutionResult(
			"ambiguous" if match.duplicate_count > 1 else "valid",
			self.profiles[match.index],
			distance=match.distance,
			raw_distance=match.raw_distance,
			mix_distance=match.mix_distance,
			total_distance=match.total_distance,
			duplicate_count=match.duplicate_count,
			resolver_time_ms=match.resolver_time_ms,
		)

	def score_profile(self, requested_counts: dict[str, Any], profile: LiveProfile) -> LiveResolutionResult:
		match = self._distance_index.distance_to_counts(requested_counts, profile.raw_counts)
		status = "invalid"
		if match.valid:
			status = "ambiguous" if match.duplicate_count > 1 else "valid"
		return LiveResolutionResult(
			status,
			profile,
			distance=match.distance,
			raw_distance=match.raw_distance,
			mix_distance=match.mix_distance,
			total_distance=match.total_distance,
			duplicate_count=match.duplicate_count,
			resolver_time_ms=match.resolver_time_ms,
		)


class ExactConfigLookup:
	def __init__(self, dataset: TuningDataset):
		self.dataset = dataset

	def resolve(self, config: dict[str, Any]) -> KernelRecord | None:
		return self.dataset.lookup_config(config)
