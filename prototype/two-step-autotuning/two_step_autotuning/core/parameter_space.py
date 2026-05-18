from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

from .dataset import TuningDataset


PARAM_INDEX_PREFIX = "idx."


@dataclass(frozen=True)
class ParameterIndexSpec:
	name: str
	values: tuple[Any, ...]


def build_parameter_index_specs(dataset: TuningDataset) -> dict[str, ParameterIndexSpec]:
	return {
		name: ParameterIndexSpec(name=name, values=tuple(values))
		for name, values in dataset.observed_parameter_values().items()
	}


def build_parameter_manipulator(specs: dict[str, ParameterIndexSpec]) -> ConfigurationManipulator:
	manipulator = ConfigurationManipulator()
	for name, spec in sorted(specs.items()):
		manipulator.add_parameter(IntegerParameter(f"{PARAM_INDEX_PREFIX}{name}", 0, len(spec.values) - 1))
	return manipulator


def config_from_tuning_indices(
	dataset: TuningDataset,
	values: dict,
	specs: dict[str, ParameterIndexSpec],
) -> dict:
	config = dict(dataset.fixed_input_config())
	for name in dataset.tuning_parameter_names():
		spec = specs[name]
		index = int(values[f"{PARAM_INDEX_PREFIX}{name}"])
		config[name] = spec.values[index]
	return config


def indices_from_config(config: dict[str, Any], specs: dict[str, ParameterIndexSpec]) -> dict[str, int]:
	indices = {}
	for name, spec in specs.items():
		value_to_index = {value: index for index, value in enumerate(spec.values)}
		indices[f"{PARAM_INDEX_PREFIX}{name}"] = value_to_index[config[name]]
	return indices
