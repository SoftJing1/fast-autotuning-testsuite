from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

from scripts.collect_tuning_performance_simple import _device_limits, _load_config_generator_modules

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


def build_generator_parameter_index_specs(
	kernel_type: str,
	input_size: str,
	device_type: str = "cpu",
) -> dict[str, ParameterIndexSpec]:
	modules = _load_config_generator_modules()
	if kernel_type not in modules:
		raise ValueError(f"Unsupported kernel type: {kernel_type}")
	dims = tuple(int(value) for value in input_size.split("x"))
	max_wi_size, max_wg_size = _device_limits(device_type)
	module = modules[kernel_type]
	if kernel_type == "gaussian":
		return _build_gaussian_generator_specs(module, dims, max_wi_size, max_wg_size)
	return _build_gemm_generator_specs(module, dims, max_wi_size, max_wg_size)


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


def config_from_parameter_indices(
	values: dict[str, Any],
	specs: dict[str, ParameterIndexSpec],
	fixed_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
	config = {} if fixed_config is None else dict(fixed_config)
	for name, spec in specs.items():
		index = int(values[f"{PARAM_INDEX_PREFIX}{name}"])
		config[name] = spec.values[index]
	return config


def indices_from_config(config: dict[str, Any], specs: dict[str, ParameterIndexSpec]) -> dict[str, int]:
	indices = {}
	for name, spec in specs.items():
		value_to_index = {value: index for index, value in enumerate(spec.values)}
		indices[f"{PARAM_INDEX_PREFIX}{name}"] = value_to_index[config[name]]
	return indices


def _build_gaussian_generator_specs(
	module,
	dims: tuple[int, ...],
	max_wi_size: tuple[int, int, int],
	max_wg_size: int,
) -> dict[str, ParameterIndexSpec]:
	h, w = dims
	dim1_values = {
		"glb_1": set(),
		"wg_1": set(),
		"lcl_1": set(),
		"wi_1": set(),
		"prv_1": set(),
	}
	dim2_values = {
		"glb_2": set(),
		"wg_2": set(),
		"lcl_2": set(),
		"wi_2": set(),
		"prv_2": set(),
	}
	for wi_1_ocl_dim, wi_2_ocl_dim in module.OCL_DIM_PAIRS:
		dim1 = module.get_valid_dim1_factors(h, max_wi_size, max_wg_size, wi_1_ocl_dim)
		for key, values in dim1.items():
			dim1_values[key.lower()].update(values)
		for wi1 in sorted(set(dim1["WI_1"])):
			dim2 = module.get_valid_dim2_factors(w, wi1, max_wi_size, max_wg_size, wi_2_ocl_dim)
			for key, values in dim2.items():
				dim2_values[key.lower()].update(values)
	return {
		"g_cb_res_dest_level": ParameterIndexSpec("g_cb_res_dest_level", (0, 1, 2)),
		"l_cb_res_dest_level": ParameterIndexSpec("l_cb_res_dest_level", (0, 1, 2)),
		"p_cb_res_dest_level": ParameterIndexSpec("p_cb_res_dest_level", (0, 1, 2)),
		"images_cache_lcl": ParameterIndexSpec("images_cache_lcl", (0, 1)),
		"images_cache_prv": ParameterIndexSpec("images_cache_prv", (0, 1)),
		"filter_cache_lcl": ParameterIndexSpec("filter_cache_lcl", (0, 1)),
		"filter_cache_prv": ParameterIndexSpec("filter_cache_prv", (0, 1)),
		"out_cache_prv": ParameterIndexSpec("out_cache_prv", (0, 1)),
		"wg_1_ocl_dim": ParameterIndexSpec("wg_1_ocl_dim", (0, 1)),
		"wg_2_ocl_dim": ParameterIndexSpec("wg_2_ocl_dim", (0, 1)),
		"wi_1_ocl_dim": ParameterIndexSpec("wi_1_ocl_dim", (0, 1)),
		"wi_2_ocl_dim": ParameterIndexSpec("wi_2_ocl_dim", (0, 1)),
		"input_size_1": ParameterIndexSpec("input_size_1", (h,)),
		"glb_1": ParameterIndexSpec("glb_1", tuple(sorted(dim1_values["glb_1"]))),
		"wg_1": ParameterIndexSpec("wg_1", tuple(sorted(dim1_values["wg_1"]))),
		"lcl_1": ParameterIndexSpec("lcl_1", tuple(sorted(dim1_values["lcl_1"]))),
		"wi_1": ParameterIndexSpec("wi_1", tuple(sorted(dim1_values["wi_1"]))),
		"prv_1": ParameterIndexSpec("prv_1", tuple(sorted(dim1_values["prv_1"]))),
		"input_size_2": ParameterIndexSpec("input_size_2", (w,)),
		"glb_2": ParameterIndexSpec("glb_2", tuple(sorted(dim2_values["glb_2"]))),
		"wg_2": ParameterIndexSpec("wg_2", tuple(sorted(dim2_values["wg_2"]))),
		"lcl_2": ParameterIndexSpec("lcl_2", tuple(sorted(dim2_values["lcl_2"]))),
		"wi_2": ParameterIndexSpec("wi_2", tuple(sorted(dim2_values["wi_2"]))),
		"prv_2": ParameterIndexSpec("prv_2", tuple(sorted(dim2_values["prv_2"]))),
	}


def _build_gemm_generator_specs(
	module,
	dims: tuple[int, ...],
	max_wi_size: tuple[int, int, int],
	max_wg_size: int,
) -> dict[str, ParameterIndexSpec]:
	m, n, k = dims
	valid_l1 = module.get_valid_dimension_factors(m, max_wi_size[0], max_wg_size)
	valid_l2 = module.get_valid_dimension_factors(n, max_wi_size[1], max_wg_size)
	wi_products = sorted(
		{
			int(wi_l1) * int(wi_l2)
			for wi_l1 in valid_l1["NUM_WI"]
			for wi_l2 in valid_l2["NUM_WI"]
		}
	)
	valid_r1_values = {
		"l_cb_size_r_1": set(),
		"num_wg_r_1": set(),
		"num_wi_r_1": set(),
		"p_cb_size_r_1": set(),
	}
	for wi_product in wi_products:
		valid_r1 = module.get_valid_dimension_factors(k, max_wi_size[2], max_wg_size, wi_product)
		valid_r1_values["l_cb_size_r_1"].update(valid_r1["L_CB_SIZE"])
		valid_r1_values["num_wg_r_1"].update(valid_r1["NUM_WG"])
		valid_r1_values["num_wi_r_1"].update(valid_r1["NUM_WI"])
		valid_r1_values["p_cb_size_r_1"].update(valid_r1["P_CB_SIZE"])
	return {
		"cache_l_cb": ParameterIndexSpec("cache_l_cb", (0, 1)),
		"cache_p_cb": ParameterIndexSpec("cache_p_cb", (0, 1)),
		"g_cb_res_dest_level": ParameterIndexSpec("g_cb_res_dest_level", (2,)),
		"l_cb_res_dest_level": ParameterIndexSpec("l_cb_res_dest_level", (0, 1, 2)),
		"p_cb_res_dest_level": ParameterIndexSpec("p_cb_res_dest_level", (0, 1, 2)),
		"ocl_dim_l_1": ParameterIndexSpec("ocl_dim_l_1", (0, 1, 2)),
		"ocl_dim_l_2": ParameterIndexSpec("ocl_dim_l_2", (0, 1, 2)),
		"ocl_dim_r_1": ParameterIndexSpec("ocl_dim_r_1", (0, 1, 2)),
		"input_size_l_1": ParameterIndexSpec("input_size_l_1", (m,)),
		"l_cb_size_l_1": ParameterIndexSpec("l_cb_size_l_1", tuple(sorted(set(valid_l1["L_CB_SIZE"])))),
		"p_cb_size_l_1": ParameterIndexSpec("p_cb_size_l_1", tuple(sorted(set(valid_l1["P_CB_SIZE"])))),
		"num_wg_l_1": ParameterIndexSpec("num_wg_l_1", tuple(sorted(set(valid_l1["NUM_WG"])))),
		"num_wi_l_1": ParameterIndexSpec("num_wi_l_1", tuple(sorted(set(valid_l1["NUM_WI"])))),
		"input_size_l_2": ParameterIndexSpec("input_size_l_2", (n,)),
		"l_cb_size_l_2": ParameterIndexSpec("l_cb_size_l_2", tuple(sorted(set(valid_l2["L_CB_SIZE"])))),
		"p_cb_size_l_2": ParameterIndexSpec("p_cb_size_l_2", tuple(sorted(set(valid_l2["P_CB_SIZE"])))),
		"num_wg_l_2": ParameterIndexSpec("num_wg_l_2", tuple(sorted(set(valid_l2["NUM_WG"])))),
		"num_wi_l_2": ParameterIndexSpec("num_wi_l_2", tuple(sorted(set(valid_l2["NUM_WI"])))),
		"input_size_r_1": ParameterIndexSpec("input_size_r_1", (k,)),
		"l_cb_size_r_1": ParameterIndexSpec("l_cb_size_r_1", tuple(sorted(valid_r1_values["l_cb_size_r_1"]))),
		"p_cb_size_r_1": ParameterIndexSpec("p_cb_size_r_1", tuple(sorted(valid_r1_values["p_cb_size_r_1"]))),
		"num_wg_r_1": ParameterIndexSpec("num_wg_r_1", tuple(sorted(valid_r1_values["num_wg_r_1"]))),
		"num_wi_r_1": ParameterIndexSpec("num_wi_r_1", tuple(sorted(valid_r1_values["num_wi_r_1"]))),
		"l_reduction": ParameterIndexSpec("l_reduction", (1,)),
		"p_write_back": ParameterIndexSpec("p_write_back", (0,)),
		"l_write_back": ParameterIndexSpec("l_write_back", (2,)),
	}
