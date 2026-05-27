from __future__ import annotations

import time
from typing import Any

import numpy as np

from .dataset import TuningDataset
from .parameter_distance import PARAMETER_DISTANCE_METRICS, ParameterDistanceModel
from .parameter_space import ParameterIndexSpec
from .types import ResolutionResult


class DatabaseApproxParameterResolver:
	"""Resolve a proposed tuning-parameter config to the nearest existing DB row."""

	def __init__(
		self,
		dataset: TuningDataset,
		specs: dict[str, ParameterIndexSpec],
		metric: str = "euclidean",
		max_distance: float | None = None,
	):
		if metric == "ordinal":
			metric = "euclidean"
		if metric not in PARAMETER_DISTANCE_METRICS:
			raise ValueError(f"Unknown parameter distance metric: {metric}")
		self.dataset = dataset
		self.metric = metric
		self.max_distance = max_distance
		self.model = ParameterDistanceModel(dataset, specs)

	def resolve(self, config: dict[str, Any]) -> ResolutionResult:
		start = time.perf_counter()
		distances = self.model.metric_distances_to_config(config, self.metric)
		best_index = int(np.argmin(distances))
		best_distance = float(distances[best_index])
		elapsed_ms = (time.perf_counter() - start) * 1000.0

		if self.max_distance is not None and best_distance > self.max_distance:
			return ResolutionResult(
				status="invalid",
				record=None,
				distance=best_distance,
				resolver_time_ms=elapsed_ms,
			)

		duplicate_count = int(np.isclose(distances, best_distance, rtol=0.0, atol=1.0e-12).sum())
		status = "ambiguous" if duplicate_count > 1 else "valid"
		return ResolutionResult(
			status=status,
			record=self.model.records[best_index],
			distance=best_distance,
			duplicate_count=duplicate_count,
			resolver_time_ms=elapsed_ms,
		)
