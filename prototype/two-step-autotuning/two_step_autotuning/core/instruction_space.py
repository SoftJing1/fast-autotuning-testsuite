from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

from .dataset import TuningDataset
from .instruction_map import select_representative_values
from .types import LiveProfile


INST_INDEX_PREFIX = "idx.inst."


@dataclass(frozen=True)
class InstructionParameterSpec:
	opcode: str
	values: tuple[int, ...]


def build_instruction_parameter_specs(
	dataset: TuningDataset,
	max_values_per_op: int,
	top_config_count: int,
	min_unique_values: int = 2,
	min_log_range: int = 1024,
	min_log_ratio: float = 16.0,
) -> dict[str, InstructionParameterSpec]:
	specs: dict[str, InstructionParameterSpec] = {}
	for op in dataset.opcodes:
		values = [int(record.raw_counts.get(op, 0)) for record in dataset.records]
		unique_values = tuple(sorted(set(values)))
		if len(unique_values) < min_unique_values:
			continue
		specs[op] = InstructionParameterSpec(opcode=op, values=unique_values)
	return specs


def build_instruction_parameter_specs_from_profiles(
	profiles: list[LiveProfile],
	max_values_per_op: int,
) -> dict[str, InstructionParameterSpec]:
	opcodes = sorted({op for profile in profiles for op in profile.raw_counts})
	specs: dict[str, InstructionParameterSpec] = {}
	for op in opcodes:
		values = [int(profile.raw_counts.get(op, 0)) for profile in profiles]
		representatives = select_representative_values(values, max_values_per_op)
		if len(representatives) > 1:
			specs[op] = InstructionParameterSpec(opcode=op, values=tuple(representatives))
	return specs


def build_instruction_manipulator(specs: dict[str, InstructionParameterSpec]) -> ConfigurationManipulator:
	manipulator = ConfigurationManipulator()
	for op in sorted(specs):
		spec = specs[op]
		manipulator.add_parameter(IntegerParameter(f"{INST_INDEX_PREFIX}{op}", 0, len(spec.values) - 1))
	return manipulator


def counts_from_tuning_indices(values: dict, specs: dict[str, InstructionParameterSpec]) -> dict[str, int]:
	counts = {}
	for op, spec in specs.items():
		index = int(values[f"{INST_INDEX_PREFIX}{op}"])
		counts[op] = int(spec.values[index])
	return counts


def indices_from_counts(
	counts: dict[str, int],
	specs: dict[str, InstructionParameterSpec],
) -> dict[str, int]:
	indices = {}
	for op, spec in specs.items():
		value_to_index = {value: index for index, value in enumerate(spec.values)}
		indices[f"{INST_INDEX_PREFIX}{op}"] = value_to_index[int(counts.get(op, 0))]
	return indices


def nearest_indices_from_counts(
	counts: dict[str, Any],
	specs: dict[str, InstructionParameterSpec],
) -> dict[str, int]:
	indices = {}
	for op, spec in specs.items():
		value = int(counts.get(op, 0))
		closest_index = min(
			range(len(spec.values)),
			key=lambda index: abs(int(spec.values[index]) - value),
		)
		indices[f"{INST_INDEX_PREFIX}{op}"] = closest_index
	return indices


def median_instruction_counts(profiles: list[LiveProfile]) -> dict[str, int]:
	opcodes = sorted({op for profile in profiles for op in profile.raw_counts})
	return {
		op: int(median(int(profile.raw_counts.get(op, 0)) for profile in profiles))
		for op in opcodes
	}
