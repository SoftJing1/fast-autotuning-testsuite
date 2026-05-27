from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import TuningDataset
from .resolver import ResolverWeights


@dataclass(frozen=True)
class InstructionNeighborResult:
	base_exp_id: int
	neighbor_exp_id: int
	distance: float
	raw_distance: float
	mix_distance: float
	total_distance: float
	runtime_ms: float
	neighbor_runtime_ms: float
	abs_runtime_diff_ms: float
	relative_runtime_diff: float
	tie_count: int
	zero_distance_neighbor: bool


class InstructionDistanceModel:
	"""Vectorized Euclidean distance model over instruction-count vectors."""

	def __init__(self, dataset: TuningDataset, weights: ResolverWeights | None = None):
		self.dataset = dataset
		self.weights = weights or ResolverWeights()
		self.opcodes = dataset.opcodes
		self.records = dataset.records
		self.exp_ids = np.array([record.exp_id for record in self.records], dtype=int)
		self.runtimes = np.array([record.runtime_ms for record in self.records], dtype=float)
		self.count_matrix = self._build_count_matrix()

	def _build_count_matrix(self) -> np.ndarray:
		return np.array(
			[[max(float(record.raw_counts.get(op, 0)), 0.0) for op in self.opcodes] for record in self.records],
			dtype=float,
		)

	def component_distances_to_counts(
		self,
		counts: dict[str, Any],
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		request_vector = np.array(
			[max(float(counts.get(op, 0)), 0.0) for op in self.opcodes],
			dtype=float,
		)
		raw_dist = np.sqrt(((self.count_matrix - request_vector) ** 2).sum(axis=1))
		mix_dist = np.zeros_like(raw_dist)
		total_dist = np.zeros_like(raw_dist)
		combined = raw_dist
		return combined, raw_dist, mix_dist, total_dist

	def nearest_neighbor(self, base_index: int) -> InstructionNeighborResult:
		combined, raw_dist, mix_dist, total_dist = self.component_distances_to_counts(
			self.records[base_index].raw_counts
		)
		combined = combined.astype(float)
		combined[base_index] = np.inf
		neighbor_index = int(np.argmin(combined))
		distance = float(combined[neighbor_index])
		tie_count = int(np.isclose(combined, distance, rtol=0.0, atol=1.0e-12).sum())
		runtime = float(self.runtimes[base_index])
		neighbor_runtime = float(self.runtimes[neighbor_index])
		abs_diff = abs(neighbor_runtime - runtime)
		relative_diff = abs_diff / runtime if runtime > 0.0 else 0.0
		return InstructionNeighborResult(
			base_exp_id=int(self.exp_ids[base_index]),
			neighbor_exp_id=int(self.exp_ids[neighbor_index]),
			distance=distance,
			raw_distance=float(raw_dist[neighbor_index]),
			mix_distance=float(mix_dist[neighbor_index]),
			total_distance=float(total_dist[neighbor_index]),
			runtime_ms=runtime,
			neighbor_runtime_ms=neighbor_runtime,
			abs_runtime_diff_ms=abs_diff,
			relative_runtime_diff=relative_diff,
			tie_count=tie_count,
			zero_distance_neighbor=bool(np.isclose(distance, 0.0, rtol=0.0, atol=1.0e-12)),
		)

	def nearest_neighbors(self, base_indices: list[int] | None = None) -> list[InstructionNeighborResult]:
		if base_indices is None:
			base_indices = list(range(len(self.records)))
		return [self.nearest_neighbor(base_index) for base_index in base_indices]


def instruction_neighbor_results_to_dicts(results: list[InstructionNeighborResult]) -> list[dict[str, Any]]:
	return [result.__dict__.copy() for result in results]
