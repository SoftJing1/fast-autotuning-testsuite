from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import TuningDataset
from .instruction_map import log1p
from .types import KernelRecord, LiveProfile, ResolutionResult


@dataclass(frozen=True)
class ResolverWeights:
	raw: float = 0.4
	normalized: float = 0.5
	total: float = 0.1


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
		self._raw_matrix: np.ndarray | None = None
		self._mix_matrix: np.ndarray | None = None
		self._total_vector: np.ndarray | None = None
		self._raw_mean: np.ndarray | None = None
		self._raw_std: np.ndarray | None = None
		self._std_raw_matrix: np.ndarray | None = None
		self._total_std = 1.0
		self.rebuild(items, opcodes)

	def rebuild(self, items: list[Any], opcodes: list[str]) -> None:
		self.items = list(items)
		self.opcodes = list(opcodes)
		if not self.items:
			self._raw_matrix = None
			return
		self._raw_matrix = np.array(
			[[log1p(item.raw_counts.get(op, 0)) for op in self.opcodes] for item in self.items],
			dtype=float,
		)
		rows = []
		for item in self.items:
			total = float(item.total_count) or 1.0
			rows.append([float(item.raw_counts.get(op, 0)) / total for op in self.opcodes])
		self._mix_matrix = np.array(rows, dtype=float)
		self._total_vector = np.log1p(np.array([item.total_count for item in self.items], dtype=float))
		self._raw_mean = self._raw_matrix.mean(axis=0)
		self._raw_std = self._raw_matrix.std(axis=0)
		self._raw_std[self._raw_std < 1.0e-9] = 1.0
		total_std = float(self._total_vector.std())
		self._total_std = total_std if total_std >= 1.0e-9 else 1.0
		self._std_raw_matrix = (self._raw_matrix - self._raw_mean) / self._raw_std

	def nearest(self, requested_counts: dict[str, Any], max_distance: float | None = None) -> _DistanceResult:
		start = time.perf_counter()
		if not self.items or self._raw_matrix is None:
			return _DistanceResult(index=None, resolver_time_ms=0.0)

		raw_vector = np.array([log1p(requested_counts.get(op, 0)) for op in self.opcodes], dtype=float)
		request_total = sum(max(float(requested_counts.get(op, 0)), 0.0) for op in self.opcodes)
		mix_denominator = request_total or 1.0
		mix_vector = np.array(
			[max(float(requested_counts.get(op, 0)), 0.0) / mix_denominator for op in self.opcodes],
			dtype=float,
		)
		total_value = np.log1p(request_total)

		std_raw = (raw_vector - self._raw_mean) / self._raw_std
		raw_dist = np.sqrt(((self._std_raw_matrix - std_raw) ** 2).mean(axis=1))
		mix_dist = np.abs(self._mix_matrix - mix_vector).sum(axis=1)
		total_dist = np.abs(self._total_vector - total_value) / self._total_std
		combined = (
			self.weights.raw * raw_dist
			+ self.weights.normalized * mix_dist
			+ self.weights.total * total_dist
		)
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
			mix_distance=float(mix_dist[best_index]),
			total_distance=float(total_dist[best_index]),
			duplicate_count=int(np.isclose(combined, best_distance, rtol=0.0, atol=1.0e-12).sum()),
			resolver_time_ms=(time.perf_counter() - start) * 1000.0,
		)

	def distance_to_counts(self, requested_counts: dict[str, Any], candidate_counts: dict[str, Any]) -> _DistanceResult:
		start = time.perf_counter()
		if self._raw_matrix is None or not self.opcodes:
			return _DistanceResult(index=None, resolver_time_ms=0.0)

		raw_vector = np.array([log1p(requested_counts.get(op, 0)) for op in self.opcodes], dtype=float)
		candidate_raw = np.array([log1p(candidate_counts.get(op, 0)) for op in self.opcodes], dtype=float)
		request_total = sum(max(float(requested_counts.get(op, 0)), 0.0) for op in self.opcodes)
		candidate_total = sum(max(float(candidate_counts.get(op, 0)), 0.0) for op in self.opcodes)
		request_mix_denominator = request_total or 1.0
		candidate_mix_denominator = candidate_total or 1.0
		request_mix = np.array(
			[max(float(requested_counts.get(op, 0)), 0.0) / request_mix_denominator for op in self.opcodes],
			dtype=float,
		)
		candidate_mix = np.array(
			[max(float(candidate_counts.get(op, 0)), 0.0) / candidate_mix_denominator for op in self.opcodes],
			dtype=float,
		)
		request_total_value = np.log1p(request_total)
		candidate_total_value = np.log1p(candidate_total)

		request_std = (raw_vector - self._raw_mean) / self._raw_std
		candidate_std = (candidate_raw - self._raw_mean) / self._raw_std
		raw_distance = float(np.sqrt(((candidate_std - request_std) ** 2).mean()))
		mix_distance = float(np.abs(candidate_mix - request_mix).sum())
		total_distance = float(abs(candidate_total_value - request_total_value) / self._total_std)
		combined_distance = (
			self.weights.raw * raw_distance
			+ self.weights.normalized * mix_distance
			+ self.weights.total * total_distance
		)
		return _DistanceResult(
			index=0,
			distance=combined_distance,
			raw_distance=raw_distance,
			mix_distance=mix_distance,
			total_distance=total_distance,
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
	):
		self.dataset = dataset
		self.weights = weights or ResolverWeights()
		self.max_distance = max_distance
		self.opcodes = dataset.opcodes
		self._records = dataset.records
		self._distance_index = _InstructionDistanceIndex(self._records, self.opcodes, self.weights)

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
		key = self.dataset.instruction_key(nearest_record.raw_counts)
		candidates = self.dataset.records_by_instruction_key.get(key, [nearest_record])
		chosen = min(candidates, key=lambda record: record.runtime_ms)
		status = "ambiguous" if len(candidates) > 1 else "valid"
		return ResolutionResult(
			status=status,
			record=chosen,
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
		self.opcodes = sorted({op for profile in self.profiles for op in profile.raw_counts})
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
