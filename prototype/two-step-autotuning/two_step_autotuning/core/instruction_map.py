from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def canonical_config_key(config: Mapping[str, Any]) -> str:
	parts = []
	for key, value in sorted(config.items()):
		if isinstance(value, bool):
			value = int(value)
		if isinstance(value, float) and value.is_integer():
			value = int(value)
		parts.append(f"{key}={value}")
	return "|".join(parts)


def sum_counts(count_maps: Iterable[Mapping[str, Any]]) -> dict[str, int]:
	total: dict[str, int] = {}
	for counts in count_maps:
		for op, value in counts.items():
			total[str(op)] = total.get(str(op), 0) + int(value)
	return total


def normalize_counts(counts: Mapping[str, Any]) -> dict[str, float]:
	total = sum(float(value) for value in counts.values())
	if total <= 0:
		return {str(key): 0.0 for key in counts}
	return {str(key): float(value) / total for key, value in counts.items()}


def select_representative_values(
	values: Iterable[int],
	max_values: int,
	priority_values: Iterable[int] = (),
) -> list[int]:
	unique = sorted({int(value) for value in values})
	if not unique or max_values <= 0:
		return []
	if len(unique) <= max_values:
		return unique

	selected = {unique[0], unique[-1], unique[len(unique) // 2]}
	for value in priority_values:
		if int(value) in unique:
			selected.add(int(value))

	if len(selected) < max_values:
		quantile_slots = max_values - len(selected)
		for slot in range(quantile_slots):
			if quantile_slots == 1:
				index = len(unique) // 2
			else:
				index = round(slot * (len(unique) - 1) / (quantile_slots - 1))
			selected.add(unique[index])
			if len(selected) >= max_values:
				break

	if len(selected) > max_values:
		# Preserve spread by re-quantizing the selected values.
		ordered = sorted(selected)
		indexes = {
			round(slot * (len(ordered) - 1) / (max_values - 1))
			for slot in range(max_values)
		}
		selected = {ordered[index] for index in indexes}

	return sorted(selected)


def log1p(value: Any) -> float:
	return math.log1p(max(float(value), 0.0))

