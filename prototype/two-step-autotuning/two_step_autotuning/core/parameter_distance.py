from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import TuningDataset
from .parameter_space import ParameterIndexSpec, indices_from_config


PARAMETER_DISTANCE_METRICS = (
	"euclidean",
)

_PARAMETER_DISTANCE_ALIASES = {
	"ordinal": "euclidean",
}


@dataclass(frozen=True)
class NearestNeighborResult:
	metric: str
	base_exp_id: int
	neighbor_exp_id: int
	distance: float
	runtime_ms: float
	neighbor_runtime_ms: float
	abs_runtime_diff_ms: float
	relative_runtime_diff: float
	tie_count: int


class ParameterDistanceModel:
	"""Vectorized distance model over existing dataset tuning parameters."""

	def __init__(self, dataset: TuningDataset, specs: dict[str, ParameterIndexSpec]):
		self.dataset = dataset
		self.specs = specs
		self.parameter_names = sorted(specs)
		self.records = dataset.records
		self.exp_ids = np.array([record.exp_id for record in self.records], dtype=int)
		self.runtimes = np.array([record.runtime_ms for record in self.records], dtype=float)
		self.index_matrix = self._build_index_matrix()

	def _build_index_matrix(self) -> np.ndarray:
		rows = []
		for record in self.records:
			encoded = indices_from_config(record.config, self.specs)
			rows.append([encoded[f"idx.{name}"] for name in self.parameter_names])
		return np.array(rows, dtype=float)

	def metric_distances_to(self, base_index: int, metric: str) -> np.ndarray:
		return self.metric_distances_to_config_vectors(
			index_vector=self.index_matrix[base_index],
			metric=metric,
		)

	def metric_distances_to_config(self, config: dict[str, Any], metric: str) -> np.ndarray:
		index_vector = self._index_vector_from_config(config)
		return self.metric_distances_to_config_vectors(index_vector, metric)

	def metric_distances_to_config_vectors(
		self,
		index_vector: np.ndarray,
		metric: str,
	) -> np.ndarray:
		metric = _PARAMETER_DISTANCE_ALIASES.get(metric, metric)
		if metric != "euclidean":
			raise ValueError(f"Unknown parameter distance metric: {metric}")
		return np.sqrt(((self.index_matrix - index_vector) ** 2).sum(axis=1))

	def _index_vector_from_config(self, config: dict[str, Any]) -> np.ndarray:
		encoded = indices_from_config(config, self.specs)
		return np.array([encoded[f"idx.{name}"] for name in self.parameter_names], dtype=float)

	def nearest_neighbor(self, base_index: int, metric: str) -> NearestNeighborResult:
		distances = self.metric_distances_to(base_index, metric).astype(float)
		distances[base_index] = np.inf
		neighbor_index = int(np.argmin(distances))
		distance = float(distances[neighbor_index])
		tie_count = int(np.isclose(distances, distance, rtol=0.0, atol=1.0e-12).sum())
		runtime = float(self.runtimes[base_index])
		neighbor_runtime = float(self.runtimes[neighbor_index])
		abs_diff = abs(neighbor_runtime - runtime)
		relative_diff = abs_diff / runtime if runtime > 0.0 else 0.0
		return NearestNeighborResult(
			metric=metric,
			base_exp_id=int(self.exp_ids[base_index]),
			neighbor_exp_id=int(self.exp_ids[neighbor_index]),
			distance=distance,
			runtime_ms=runtime,
			neighbor_runtime_ms=neighbor_runtime,
			abs_runtime_diff_ms=abs_diff,
			relative_runtime_diff=relative_diff,
			tie_count=tie_count,
		)

	def nearest_neighbors(
		self,
		metric: str,
		base_indices: list[int] | None = None,
	) -> list[NearestNeighborResult]:
		if base_indices is None:
			base_indices = list(range(len(self.records)))
		return [self.nearest_neighbor(base_index, metric) for base_index in base_indices]


def neighbor_results_to_dicts(results: list[NearestNeighborResult]) -> list[dict[str, Any]]:
	return [result.__dict__.copy() for result in results]
