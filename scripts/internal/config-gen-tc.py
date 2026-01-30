"""Generate tuned versions of Tensor Contraction (TC) kernels using OpenTuner.

This script provides:
1. Parameter definition separated from compilation logic
2. Tuned kernel generation by feeding parameters to the compiler
3. Exhaustive configuration iteration through all possible valid configurations
4. CLI for fast testing and batch generation

Usage:
    python -m scripts.generate_tc --help
    python -m scripts.generate_tc list-params
    python -m scripts.generate_tc generate --config config.json --output kernel.cl
    python -m scripts.generate_tc exhaustive --input-size 64 64 64 64 64 64 64 --output-dir ./kernels/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from itertools import product, permutations
from pathlib import Path
from typing import Dict, Iterator, Tuple

try:
    from opentuner import ConfigurationManipulator, IntegerParameter
except ImportError:
    print("Error: opentuner not installed. Install with: pip install opentuner", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# PARAMETER DEFINITION
# =============================================================================

def get_parameters_definition(M1: int, M2: int, M3: int, M4: int, M5: int, M6: int, N1: int) -> ConfigurationManipulator:
    """Define the OpenTuner parameter search space for TC kernel tuning.
    
    This function encapsulates all tuning parameters and their constraints
    based on the OpenTuner example in tc.cpp.
    
    Args:
        M1-M6: M tensor dimensions
        N1: N tensor dimension (reduction dimension)
    
    Returns:
        ConfigurationManipulator with all parameters and ranges defined
    """
    manipulator = ConfigurationManipulator()

    # Cache parameters
    manipulator.add_parameter(IntegerParameter('CACHE_L_CB', 0, 1))
    manipulator.add_parameter(IntegerParameter('CACHE_P_CB', 0, 1))
    manipulator.add_parameter(IntegerParameter('G_CB_RES_DEST_LEVEL', 2, 2))
    manipulator.add_parameter(IntegerParameter('L_CB_RES_DEST_LEVEL', 0, 2))
    manipulator.add_parameter(IntegerParameter('P_CB_RES_DEST_LEVEL', 0, 2))

    # OCL dimensions - 7D work space, all dimensions must be unique
    for i in range(1, 7):
        manipulator.add_parameter(IntegerParameter(f'OCL_DIM_L_{i}', 0, 6))
    manipulator.add_parameter(IntegerParameter('OCL_DIM_R_1', 0, 6))

    # L_1 (M1) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_1', M1, M1))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_1', 1, M1))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_1', 1, M1))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_1', 1, M1))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_1', 1, M1))

    # L_2 (M2) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_2', M2, M2))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_2', 1, M2))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_2', 1, M2))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_2', 1, M2))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_2', 1, M2))

    # L_3 (M3) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_3', M3, M3))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_3', 1, M3))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_3', 1, M3))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_3', 1, M3))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_3', 1, M3))

    # L_4 (M4) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_4', M4, M4))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_4', 1, M4))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_4', 1, M4))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_4', 1, M4))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_4', 1, M4))

    # L_5 (M5) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_5', M5, M5))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_5', 1, M5))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_5', 1, M5))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_5', 1, M5))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_5', 1, M5))

    # L_6 (M6) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_6', M6, M6))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_6', 1, M6))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_6', 1, M6))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_6', 1, M6))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_6', 1, M6))

    # R_1 (N1) dimension parameters - reduction dimension
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_R_1', N1, N1))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_R_1', 1, N1))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_R_1', 1, N1))
    manipulator.add_parameter(IntegerParameter('NUM_WG_R_1', 1, N1))
    manipulator.add_parameter(IntegerParameter('NUM_WI_R_1', 1, N1))

    # Fixed parameters
    manipulator.add_parameter(IntegerParameter('L_REDUCTION', 1, 1))
    manipulator.add_parameter(IntegerParameter('P_WRITE_BACK', 0, 0))
    manipulator.add_parameter(IntegerParameter('L_WRITE_BACK', 6, 6))

    return manipulator


def get_parameters_info() -> Dict:
    """Return a dictionary with metadata about all tuning parameters.
    
    Returns:
        Dict with parameter names, ranges, and descriptions
    """
    return {
        "cache_parameters": {
            "CACHE_L_CB": "Cache at local level for L tensors",
            "CACHE_P_CB": "Cache at private level",
            "G_CB_RES_DEST_LEVEL": "Global cache resolution destination level",
            "L_CB_RES_DEST_LEVEL": "Local cache resolution destination level",
            "P_CB_RES_DEST_LEVEL": "Private cache resolution destination level",
        },
        "ocl_dimensions": {
            "OCL_DIM_L_1": "OpenCL dimension for L_1 (M1) in {0..6}",
            "OCL_DIM_L_2": "OpenCL dimension for L_2 (M2) in {0..6}",
            "OCL_DIM_L_3": "OpenCL dimension for L_3 (M3) in {0..6}",
            "OCL_DIM_L_4": "OpenCL dimension for L_4 (M4) in {0..6}",
            "OCL_DIM_L_5": "OpenCL dimension for L_5 (M5) in {0..6}",
            "OCL_DIM_L_6": "OpenCL dimension for L_6 (M6) in {0..6}",
            "OCL_DIM_R_1": "OpenCL dimension for R_1 (N1) in {0..6}",
        },
        "dimension_parameters": {
            "L_1": ["INPUT_SIZE_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1", "NUM_WG_L_1", "NUM_WI_L_1"],
            "L_2": ["INPUT_SIZE_L_2", "L_CB_SIZE_L_2", "P_CB_SIZE_L_2", "NUM_WG_L_2", "NUM_WI_L_2"],
            "L_3": ["INPUT_SIZE_L_3", "L_CB_SIZE_L_3", "P_CB_SIZE_L_3", "NUM_WG_L_3", "NUM_WI_L_3"],
            "L_4": ["INPUT_SIZE_L_4", "L_CB_SIZE_L_4", "P_CB_SIZE_L_4", "NUM_WG_L_4", "NUM_WI_L_4"],
            "L_5": ["INPUT_SIZE_L_5", "L_CB_SIZE_L_5", "P_CB_SIZE_L_5", "NUM_WG_L_5", "NUM_WI_L_5"],
            "L_6": ["INPUT_SIZE_L_6", "L_CB_SIZE_L_6", "P_CB_SIZE_L_6", "NUM_WG_L_6", "NUM_WI_L_6"],
            "R_1": ["INPUT_SIZE_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1", "NUM_WG_R_1", "NUM_WI_R_1"],
        },
    }


# =============================================================================
# CONFIGURATION VALIDATION AND CONSTRAINTS
# =============================================================================

def is_configuration_valid(
    config: Dict,
    max_wi_size: Tuple[int, int, int],
    max_wg_size: int
) -> bool:
    """Validate a configuration against hardware and structural constraints.
    
    Constraints are based on the validation logic from tc.cpp:
    - Work item sizes must not exceed hardware limits per dimension
    - All 7 dimensions (6 L + 1 R) must be unique
    - Divisibility constraints for each dimension
    
    Args:
        config: Dictionary with parameter values
        max_wi_size: Tuple of max work item sizes per dimension
        max_wg_size: Max work group size
    
    Returns:
        True if configuration is valid, False otherwise
    """
    cfg = config

    # All OCL dimensions must be unique
    ocl_dims = {
        cfg['OCL_DIM_L_1'], cfg['OCL_DIM_L_2'], cfg['OCL_DIM_L_3'],
        cfg['OCL_DIM_L_4'], cfg['OCL_DIM_L_5'], cfg['OCL_DIM_L_6'],
        cfg['OCL_DIM_R_1']
    }
    if len(ocl_dims) != 7:
        return False

    # Calculate total work items per dimension
    num_wi = [1, 1, 1]
    
    for i in range(1, 7):
        dim_idx = cfg[f'OCL_DIM_L_{i}']
        num_wi[dim_idx if dim_idx < 3 else 2] *= cfg[f'NUM_WI_L_{i}']
    
    r_dim_idx = cfg['OCL_DIM_R_1']
    num_wi[r_dim_idx if r_dim_idx < 3 else 2] *= cfg['NUM_WI_R_1']

    # Check hardware constraints
    for i in range(3):
        if num_wi[i] > max_wi_size[i]:
            return False
    if num_wi[0] * num_wi[1] * num_wi[2] > max_wg_size:
        return False

    # Check divisibility constraints for each dimension
    for i in range(1, 7):
        INPUT_SIZE = cfg[f'INPUT_SIZE_L_{i}']
        L_CB_SIZE = cfg[f'L_CB_SIZE_L_{i}']
        NUM_WG = cfg[f'NUM_WG_L_{i}']
        NUM_WI = cfg[f'NUM_WI_L_{i}']
        P_CB_SIZE = cfg[f'P_CB_SIZE_L_{i}']

        # Prevent zero cached-iteration counts in generated kernels
        if L_CB_SIZE < NUM_WI:
            return False
        
        if INPUT_SIZE % L_CB_SIZE != 0:
            return False
        rem = INPUT_SIZE // L_CB_SIZE
        if rem % NUM_WG != 0:
            return False
        rem = rem // NUM_WG
        if rem % NUM_WI != 0:
            return False
        rem = rem // NUM_WI
        if rem != P_CB_SIZE:
            return False

    # Check R_1 divisibility
    INPUT_SIZE = cfg['INPUT_SIZE_R_1']
    L_CB_SIZE = cfg['L_CB_SIZE_R_1']
    NUM_WG = cfg['NUM_WG_R_1']
    NUM_WI = cfg['NUM_WI_R_1']
    P_CB_SIZE = cfg['P_CB_SIZE_R_1']

    # Prevent zero cached-iteration counts in generated kernels
    if L_CB_SIZE < NUM_WI:
        return False
    
    if INPUT_SIZE % L_CB_SIZE != 0:
        return False
    rem = INPUT_SIZE // L_CB_SIZE
    if rem % NUM_WG != 0:
        return False
    rem = rem // NUM_WG
    if rem % NUM_WI != 0:
        return False
    rem = rem // NUM_WI
    if rem != P_CB_SIZE:
        return False

    return True


# =============================================================================
# KERNEL GENERATION
# =============================================================================

def get_kernel_template_path() -> str:
    """Get the path to the TC kernel template.
    
    Returns:
        Absolute path to tc_abcdef_gebc_dfga_1.cl
    """
    script_dir = Path(__file__).parent
    kernel_dir = script_dir.parent / "kernels"
    template_path = kernel_dir / "kernel-template" / "tc_abcdef_gebc_dfga_1.cl"
    return str(template_path.absolute())


def generate_tuned_kernel(
    config: Dict,
    output_path: str,
    template_path: str | None = None,
    device_type: str = "cpu"
) -> bool:
    """Generate a tuned kernel by feeding parameters to the compiler.
    
    Args:
        config: Dictionary with parameter names and values
        output_path: Where to write the generated kernel
        template_path: Path to kernel template (defaults to tc_abcdef_gebc_dfga_1.cl)
        device_type: Device type ("cpu" or "gpu") - used for validation
    
    Returns:
        True if generation succeeded, False otherwise
    """
    if template_path is None:
        template_path = get_kernel_template_path()

    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}", file=sys.stderr)
        return False

    # Set hardware limits based on device type
    if device_type == "cpu":
        max_wi_size = (8192, 8192, 8192)
        max_wg_size = 8192
    else:
        max_wi_size = (1024, 1024, 64)
        max_wg_size = 1024

    # Validate configuration
    if not is_configuration_valid(config, max_wi_size, max_wg_size):
        print(f"Warning: Invalid configuration", file=sys.stderr)
        return False

    # Read template
    try:
        with open(template_path, 'r') as f:
            kernel_source = f.read()
    except IOError as e:
        print(f"Error reading template: {e}", file=sys.stderr)
        return False

    # Add configuration parameters as preprocessor directives
    defines = "\n".join([f"#define {k} {v}" for k, v in sorted(config.items())])
    output_source = defines + "\n\n" + kernel_source

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            f.write(output_source)
        return True
    except IOError as e:
        print(f"Error writing kernel: {e}", file=sys.stderr)
        return False


def save_tuning_parameters_only(
    config: Dict,
    output_path: str,
    device_type: str = "cpu"
) -> bool:
    """Save only the tuning parameter definitions without the template.
    
    Args:
        config: Dictionary with parameter names and values
        output_path: Where to write the parameter definitions
        device_type: Device type ("cpu" or "gpu") - used for validation
    
    Returns:
        True if save succeeded, False otherwise
    """
    # Set hardware limits based on device type
    if device_type == "cpu":
        max_wi_size = (8192, 8192, 8192)
        max_wg_size = 8192
    else:
        max_wi_size = (1024, 1024, 64)
        max_wg_size = 1024

    # Validate configuration
    if not is_configuration_valid(config, max_wi_size, max_wg_size):
        print(f"Warning: Invalid configuration", file=sys.stderr)
        return False

    # Create parameter definitions only
    defines = "\n".join([f"#define {k} {v}" for k, v in sorted(config.items())])

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            f.write(defines + "\n")
        return True
    except IOError as e:
        print(f"Error writing parameters: {e}", file=sys.stderr)
        return False


# =============================================================================
# CONSTRAINT-AWARE CONFIGURATION GENERATION
# =============================================================================

def get_valid_dimension_factors(
    INPUT_SIZE: int,
    max_wg_size: int,
) -> Dict[str, list]:
    """Generate all valid dimension factor combinations respecting divisibility.
    
    Args:
        INPUT_SIZE: Input size for this dimension
        max_wg_size: Max work group size
    
    Returns:
        Dict mapping parameter names to lists of valid values
    """
    result = {
        'L_CB_SIZE': [],
        'NUM_WG': [],
        'NUM_WI': [],
        'P_CB_SIZE': [],
    }
    
    valid_tuples = []
    
    # Divisors of INPUT_SIZE
    lcb_options = [i for i in range(1, INPUT_SIZE + 1) if INPUT_SIZE % i == 0]
    
    for lcb in lcb_options:
        rem1 = INPUT_SIZE // lcb
        wg_options = [i for i in range(1, rem1 + 1) if rem1 % i == 0]
        
        for wg in wg_options:
            rem2 = rem1 // wg
            wi_options = [i for i in range(1, rem2 + 1) if rem2 % i == 0]
            
            for wi in wi_options:
                pcb = rem2 // wi
                valid_tuples.append((lcb, wg, wi, pcb))
    
    if valid_tuples:
        for lcb, wg, wi, pcb in valid_tuples:
            result['L_CB_SIZE'].append(lcb)
            result['NUM_WG'].append(wg)
            result['NUM_WI'].append(wi)
            result['P_CB_SIZE'].append(pcb)
    
    return result


# =============================================================================
# EXHAUSTIVE CONFIGURATION ITERATION
# =============================================================================

def exhaustive_iterate_configurations(
    M1: int, M2: int, M3: int, M4: int, M5: int, M6: int, N1: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024
) -> Iterator[Dict]:
    """Exhaustively iterate through all valid configurations.
    
    Args:
        M1-M6: M tensor dimensions
        N1: N tensor (reduction) dimension
        max_wi_size: Max work item sizes per dimension
        max_wg_size: Max work group size
    
    Yields:
        Valid configuration dictionaries
    """
    # Fixed parameters
    fixed_params = {
        'G_CB_RES_DEST_LEVEL': 2,
        'L_REDUCTION': 1,
        'P_WRITE_BACK': 0,
        'L_WRITE_BACK': 6,
        'INPUT_SIZE_L_1': M1,
        'INPUT_SIZE_L_2': M2,
        'INPUT_SIZE_L_3': M3,
        'INPUT_SIZE_L_4': M4,
        'INPUT_SIZE_L_5': M5,
        'INPUT_SIZE_L_6': M6,
        'INPUT_SIZE_R_1': N1,
    }

    # Variable cache parameters
    cache_params = {
        'CACHE_L_CB': [0, 1],
        'CACHE_P_CB': [0, 1],
    }

    # Cache level parameters with constraints
    cb_res_levels = {
        'L_CB_RES_DEST_LEVEL': [0, 1, 2],
        'P_CB_RES_DEST_LEVEL': [0, 1, 2],
    }

    # Pre-compute valid dimension factors
    valid_dims = {}
    for i, size in enumerate([M1, M2, M3, M4, M5, M6, N1], 1):
        dim_name = f'L_{i}' if i <= 6 else 'R_1'
        valid_dims[dim_name] = get_valid_dimension_factors(size, max_wg_size)

    cache_flag_names = list(cache_params.keys())
    cache_flag_values = [cache_params[name] for name in cache_flag_names]
    
    cb_res_names = list(cb_res_levels.keys())
    cb_res_values = [cb_res_levels[name] for name in cb_res_names]

    # Generate all permutations of 7 unique dimensions
    all_perms = list(permutations(range(7)))

    # Iterate through cache and CB resolution combinations
    for cache_combo in product(*cache_flag_values):
        cache_dict = dict(zip(cache_flag_names, cache_combo))
        
        for cb_res_combo in product(*cb_res_values):
            # Check hierarchy constraint
            if cb_res_combo[1] > cb_res_combo[0]:
                continue
            
            cb_res_dict = dict(zip(cb_res_names, cb_res_combo))

            # Iterate through OCL dimension permutations
            for perm in all_perms:
                ocl_dict = {
                    'OCL_DIM_L_1': perm[0],
                    'OCL_DIM_L_2': perm[1],
                    'OCL_DIM_L_3': perm[2],
                    'OCL_DIM_L_4': perm[3],
                    'OCL_DIM_L_5': perm[4],
                    'OCL_DIM_L_6': perm[5],
                    'OCL_DIM_R_1': perm[6],
                }

                # Generate all combinations of dimension factors
                dim_factor_lists = []
                for i in range(1, 7):
                    dim_name = f'L_{i}'
                    valid = valid_dims[dim_name]
                    dim_factor_lists.append((i, valid))
                
                # Add R_1
                dim_factor_lists.append(('R_1', valid_dims['R_1']))

                # Iterate through all factor combinations
                def iterate_factor_combinations(dim_list_idx, current_dict):
                    if dim_list_idx >= len(dim_factor_lists):
                        # All dimensions processed
                        config = {**fixed_params, **cache_dict, **cb_res_dict, **ocl_dict, **current_dict}
                        if is_configuration_valid(config, max_wi_size, max_wg_size):
                            yield config
                        return
                    
                    dim_num, valid = dim_factor_lists[dim_list_idx]
                    dim_prefix = f'L_{dim_num}' if isinstance(dim_num, int) else 'R_1'
                    
                    for idx in range(len(valid['L_CB_SIZE'])):
                        new_dict = {
                            **current_dict,
                            f'L_CB_SIZE_{dim_prefix}': valid['L_CB_SIZE'][idx],
                            f'NUM_WG_{dim_prefix}': valid['NUM_WG'][idx],
                            f'NUM_WI_{dim_prefix}': valid['NUM_WI'][idx],
                            f'P_CB_SIZE_{dim_prefix}': valid['P_CB_SIZE'][idx],
                        }
                        yield from iterate_factor_combinations(dim_list_idx + 1, new_dict)

                yield from iterate_factor_combinations(0, {})


def count_valid_configurations(
    M1: int, M2: int, M3: int, M4: int, M5: int, M6: int, N1: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024
) -> int:
    """Count the total number of valid configurations without storing them."""
    count = 0
    for _ in exhaustive_iterate_configurations(M1, M2, M3, M4, M5, M6, N1, max_wi_size, max_wg_size):
        count += 1
    return count


def random_sample_configurations(
    M1: int, M2: int, M3: int, M4: int, M5: int, M6: int, N1: int,
    num_samples: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024,
    seed: int | None = None,
    attempts_per_sample: int = 1000,
) -> Iterator[Dict]:
    """Randomly generate valid configurations without full enumeration."""
    if seed is not None:
        random.seed(seed)

    fixed_params = {
        'G_CB_RES_DEST_LEVEL': 2,
        'L_REDUCTION': 1,
        'P_WRITE_BACK': 0,
        'L_WRITE_BACK': 6,
        'INPUT_SIZE_L_1': M1,
        'INPUT_SIZE_L_2': M2,
        'INPUT_SIZE_L_3': M3,
        'INPUT_SIZE_L_4': M4,
        'INPUT_SIZE_L_5': M5,
        'INPUT_SIZE_L_6': M6,
        'INPUT_SIZE_R_1': N1,
    }

    cache_params = {
        'CACHE_L_CB': [0, 1],
        'CACHE_P_CB': [0, 1],
    }

    cb_res_levels = {
        'L_CB_RES_DEST_LEVEL': [0, 1, 2],
        'P_CB_RES_DEST_LEVEL': [0, 1, 2],
    }

    # Pre-compute valid dimension factors
    valid_dims = {}
    for i, size in enumerate([M1, M2, M3, M4, M5, M6, N1], 1):
        dim_name = f'L_{i}' if i <= 6 else 'R_1'
        valid_dims[dim_name] = get_valid_dimension_factors(size, max_wg_size)

    cache_flag_names = list(cache_params.keys())
    cache_flag_values = [cache_params[name] for name in cache_flag_names]
    
    cb_res_names = list(cb_res_levels.keys())
    cb_res_values = [cb_res_levels[name] for name in cb_res_names]

    all_perms = list(permutations(range(7)))

    seen = set()
    generated = 0
    valid_count = 0

    # Generate samples until N valid kernels are produced
    while valid_count < num_samples:

        # Random cache flags
        cache_combo = tuple(random.choice(v) for v in cache_flag_values)
        cache_dict = dict(zip(cache_flag_names, cache_combo))

        # Random CB resolution levels
        l_cb_res = random.choice(cb_res_levels['L_CB_RES_DEST_LEVEL'])
        p_cb_res = random.choice([v for v in cb_res_levels['P_CB_RES_DEST_LEVEL'] if v <= l_cb_res])
        cb_res_dict = {
            'L_CB_RES_DEST_LEVEL': l_cb_res,
            'P_CB_RES_DEST_LEVEL': p_cb_res,
        }

        # Random OCL dimensions
        perm = random.choice(all_perms)
        ocl_dict = {
            'OCL_DIM_L_1': perm[0],
            'OCL_DIM_L_2': perm[1],
            'OCL_DIM_L_3': perm[2],
            'OCL_DIM_L_4': perm[3],
            'OCL_DIM_L_5': perm[4],
            'OCL_DIM_L_6': perm[5],
            'OCL_DIM_R_1': perm[6],
        }

        # Random dimension factor choices
        factor_dict = {}
        for i in range(1, 7):
            dim_name = f'L_{i}'
            valid = valid_dims[dim_name]
            idx = random.randint(0, len(valid['L_CB_SIZE']) - 1)
            factor_dict[f'L_CB_SIZE_{dim_name}'] = valid['L_CB_SIZE'][idx]
            factor_dict[f'NUM_WG_{dim_name}'] = valid['NUM_WG'][idx]
            factor_dict[f'NUM_WI_{dim_name}'] = valid['NUM_WI'][idx]
            factor_dict[f'P_CB_SIZE_{dim_name}'] = valid['P_CB_SIZE'][idx]
        
        # R_1
        valid_r1 = valid_dims['R_1']
        idx = random.randint(0, len(valid_r1['L_CB_SIZE']) - 1)
        factor_dict[f'L_CB_SIZE_R_1'] = valid_r1['L_CB_SIZE'][idx]
        factor_dict[f'NUM_WG_R_1'] = valid_r1['NUM_WG'][idx]
        factor_dict[f'NUM_WI_R_1'] = valid_r1['NUM_WI'][idx]
        factor_dict[f'P_CB_SIZE_R_1'] = valid_r1['P_CB_SIZE'][idx]

        config = {**fixed_params, **cache_dict, **cb_res_dict, **ocl_dict, **factor_dict}

        # De-duplication
        key = tuple(config.items())
        if key in seen:
            continue

        seen.add(key)
        generated += 1
        if is_configuration_valid(config, max_wi_size, max_wg_size):
            valid_count += 1
            
        yield config


# =============================================================================
# CLI
# =============================================================================

def cmd_list_params(args) -> int:
    """List all tuning parameters and their descriptions."""
    info = get_parameters_info()
    print(json.dumps(info, indent=2))
    return 0


def cmd_list_params_for_size(args) -> int:
    """List parameter definitions for a specific input size."""
    sizes = args.input_size  # [M1, M2, M3, M4, M5, M6, N1]
    manipulator = get_parameters_definition(*sizes)

    # Extract parameter info from manipulator
    params_info = {}
    for param in manipulator.parameters:
        params_info[param.name] = {
            "min": param.min_value,
            "max": param.max_value,
            "type": "integer"
        }

    print(json.dumps(params_info, indent=2))
    return 0


def cmd_generate(args) -> int:
    """Generate a single tuned kernel from a configuration file."""
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    device_type = args.device_type or "gpu"
    success = generate_tuned_kernel(config, args.output, device_type=device_type)
    if success:
        print(f"Generated kernel: {args.output}")
        return 0
    else:
        return 1


def cmd_exhaustive(args) -> int:
    """Exhaustively generate all valid configurations."""
    sizes = args.input_size  # [M1, M2, M3, M4, M5, M6, N1]

    # Determine device limits
    if args.device_type == "cpu":
        max_wi_size = (8192, 8192, 8192)
        max_wg_size = 8192
    else:
        max_wi_size = (1024, 1024, 64)
        max_wg_size = 1024

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.count_only:
        count = count_valid_configurations(*sizes, max_wi_size, max_wg_size)
        print(f"Total valid configurations: {count}")
        return 0

    # Choose iteration strategy
    if args.random:
        num_samples = args.random
        seed = args.seed
        attempts_per_sample = args.random_attempts_per_sample
        config_iter = random_sample_configurations(
            *sizes, num_samples, max_wi_size, max_wg_size, seed, attempts_per_sample
        )
        if args.verbose:
            print(
                f"Sampling {num_samples} random configurations (seed={seed}, "
                f"attempts/sample={attempts_per_sample})"
            )
    else:
        config_iter = exhaustive_iterate_configurations(*sizes, max_wi_size, max_wg_size)
        if args.verbose and args.max:
            print(f"Generating first {args.max} configurations sequentially")

    # Generate kernels
    size_str = "x".join(str(s) for s in sizes)
    generated = 0
    invalid = 0
    for i, config in enumerate(config_iter):
        if args.max and generated >= args.max:
            break

        output_file = os.path.join(output_dir, f"tc_{size_str}_{i:06d}.cl")

        if save_tuning_parameters_only(config, output_file, device_type=args.device_type):
            generated += 1
            if args.verbose:
                print(f"[{generated}] Generated: {output_file}")
        else:
            invalid += 1
            if args.verbose:
                print(f"[SKIP] Failed to generate: {output_file}")

    print(f"Generated: {generated} kernels")
    if invalid > 0:
        print(f"Failed: {invalid} kernels")

    return 0


def cmd_validate(args) -> int:
    """Validate a configuration file."""
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    device_type = args.device_type or "gpu"
    if device_type == "cpu":
        max_wi_size = (8192, 8192, 8192)
        max_wg_size = 8192
    else:
        max_wi_size = (1024, 1024, 64)
        max_wg_size = 1024

    if is_configuration_valid(config, max_wi_size, max_wg_size):
        print("Configuration is VALID")
        print(json.dumps(config, indent=2))
        return 0
    else:
        print("Configuration is INVALID")
        print(json.dumps(config, indent=2))
        return 1


def main():
    """Parse arguments and dispatch to appropriate command."""
    parser = argparse.ArgumentParser(
        description="Generate tuned TC (Tensor Contraction) kernels with different parameter configurations"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # list-params command
    subparsers.add_parser(
        'list-params',
        help='List all available tuning parameters'
    ).set_defaults(func=cmd_list_params)

    # list-params-for-size command
    parser_size = subparsers.add_parser(
        'list-params-for-size',
        help='List parameter definitions for a specific input size'
    )
    parser_size.add_argument('input_size', type=int, nargs=7, metavar=('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'N1'),
                             help='Input dimensions (M1 M2 M3 M4 M5 M6 N1)')
    parser_size.set_defaults(func=cmd_list_params_for_size)

    # generate command
    parser_gen = subparsers.add_parser(
        'generate',
        help='Generate a single tuned kernel from configuration'
    )
    parser_gen.add_argument('--config', required=True, help='Path to config JSON file')
    parser_gen.add_argument('--output', required=True, help='Output kernel file path')
    parser_gen.add_argument('--device-type', choices=['cpu', 'gpu'], default='gpu',
                            help='Device type for hardware limits')
    parser_gen.set_defaults(func=cmd_generate)

    # exhaustive command
    parser_exh = subparsers.add_parser(
        'exhaustive',
        help='Exhaustively generate all valid configurations or random sample'
    )
    parser_exh.add_argument('input_size', type=int, nargs=7, metavar=('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'N1'),
                            help='Input dimensions (M1 M2 M3 M4 M5 M6 N1)')
    parser_exh.add_argument('--output-dir', required=True, help='Output directory')
    parser_exh.add_argument('--device-type', choices=['cpu', 'gpu'], default='gpu',
                            help='Device type for hardware limits')
    parser_exh.add_argument('--max', type=int, help='Maximum number of kernels to generate sequentially')
    parser_exh.add_argument('--random', type=int, metavar='N',
                            help='Generate N random configurations instead of exhaustive')
    parser_exh.add_argument('--random-attempts-per-sample', type=int, default=1000,
                            help='Rejection sampling attempt budget per sample (default: 1000)')
    parser_exh.add_argument('--seed', type=int, default=None,
                            help='Random seed for reproducible random sampling')
    parser_exh.add_argument('--count-only', action='store_true',
                            help='Only count valid configurations, do not generate')
    parser_exh.add_argument('--verbose', action='store_true',
                            help='Print detailed progress')
    parser_exh.set_defaults(func=cmd_exhaustive)

    # validate command
    parser_val = subparsers.add_parser(
        'validate',
        help='Validate a configuration file'
    )
    parser_val.add_argument('--config', required=True, help='Path to config JSON file')
    parser_val.add_argument('--device-type', choices=['cpu', 'gpu'], default='gpu',
                            help='Device type for hardware limits')
    parser_val.set_defaults(func=cmd_validate)

    # Parse arguments
    args = parser.parse_args()

    # Show help if no command
    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
