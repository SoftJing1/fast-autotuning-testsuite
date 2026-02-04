"""Generate tuned versions of Gaussian kernels using OpenTuner.

This script provides:
1. Parameter definition separated from compilation logic
2. Tuned kernel generation by feeding parameters to the compiler
3. Exhaustive configuration iteration through all possible valid configurations
4. CLI for fast testing and batch generation

Usage:
    python -m scripts.generate_gaussian --help
    python -m scripts.generate_gaussian list-params
    python -m scripts.generate_gaussian generate --config config.json --output kernel.cl
    python -m scripts.generate_gaussian exhaustive --input-size 1024 1024 --output-dir ./kernels/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from itertools import product
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

def get_parameters_definition(H: int, W: int) -> ConfigurationManipulator:
    """Define the OpenTuner parameter search space for Gaussian kernel tuning.
    
    This function encapsulates all tuning parameters and their constraints
    based on the OpenTuner example in gaussian.py.
    
    Args:
        H: Height of the input (already adjusted, i.e., actual size - 4)
        W: Width of the input (already adjusted, i.e., actual size - 4)
    
    Returns:
        ConfigurationManipulator with all parameters and ranges defined
    """
    manipulator = ConfigurationManipulator()

    # Cache level parameters - control where data is cached
    manipulator.add_parameter(IntegerParameter('G_CB_RES_DEST_LEVEL', 2, 2))
    manipulator.add_parameter(IntegerParameter('L_CB_RES_DEST_LEVEL', 0, 0))
    manipulator.add_parameter(IntegerParameter('P_CB_RES_DEST_LEVEL', 0, 0))

    # Cache flags - enable/disable caching at different levels
    manipulator.add_parameter(IntegerParameter('IMAGES_CACHE_LCL', 0, 1))
    manipulator.add_parameter(IntegerParameter('IMAGES_CACHE_PRV', 0, 1))
    manipulator.add_parameter(IntegerParameter('FILTER_CACHE_LCL', 0, 1))
    manipulator.add_parameter(IntegerParameter('FILTER_CACHE_PRV', 0, 1))
    manipulator.add_parameter(IntegerParameter('OUT_CACHE_PRV', 0, 1))

    # Work group dimensions
    manipulator.add_parameter(IntegerParameter('WG_1_OCL_DIM', 1, 1))
    manipulator.add_parameter(IntegerParameter('WG_2_OCL_DIM', 0, 0))

    # Work item dimensions
    manipulator.add_parameter(IntegerParameter('WI_1_OCL_DIM', 1, 1))
    manipulator.add_parameter(IntegerParameter('WI_2_OCL_DIM', 0, 0))

    # Dimension 1 parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_1', H, H))
    manipulator.add_parameter(IntegerParameter('GLB_1', 1, H))
    manipulator.add_parameter(IntegerParameter('WG_1', 1, H))
    manipulator.add_parameter(IntegerParameter('LCL_1', 1, H))
    manipulator.add_parameter(IntegerParameter('WI_1', 1, H))
    manipulator.add_parameter(IntegerParameter('PRV_1', 1, H))

    # Dimension 2 parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_2', W, W))
    manipulator.add_parameter(IntegerParameter('GLB_2', 1, W))
    manipulator.add_parameter(IntegerParameter('WG_2', 1, W))
    manipulator.add_parameter(IntegerParameter('LCL_2', 1, W))
    manipulator.add_parameter(IntegerParameter('WI_2', 1, W))
    manipulator.add_parameter(IntegerParameter('PRV_2', 1, W))

    return manipulator


def get_parameters_info() -> Dict:
    """Return a dictionary with metadata about all tuning parameters.
    
    Returns:
        Dict with parameter names, ranges, and descriptions
    """
    return {
        "cache_level_parameters": {
            "G_CB_RES_DEST_LEVEL": "Global cache resolution destination level",
            "L_CB_RES_DEST_LEVEL": "Local cache resolution destination level",
            "P_CB_RES_DEST_LEVEL": "Private cache resolution destination level",
        },
        "cache_flags": {
            "IMAGES_CACHE_LCL": "Cache images in local memory",
            "IMAGES_CACHE_PRV": "Cache images in private memory",
            "FILTER_CACHE_LCL": "Cache filter in local memory",
            "FILTER_CACHE_PRV": "Cache filter in private memory",
            "OUT_CACHE_PRV": "Cache output in private memory",
        },
        "work_group_dimensions": {
            "WG_1_OCL_DIM": "Work group dimension 1 (0=x, 1=y)",
            "WG_2_OCL_DIM": "Work group dimension 2 (0=x, 1=y)",
        },
        "work_item_dimensions": {
            "WI_1_OCL_DIM": "Work item dimension 1 (0=x, 1=y)",
            "WI_2_OCL_DIM": "Work item dimension 2 (0=x, 1=y)",
        },
        "dimension_1_parameters": [
            "INPUT_SIZE_1", "GLB_1", "WG_1", "LCL_1", "WI_1", "PRV_1"
        ],
        "dimension_2_parameters": [
            "INPUT_SIZE_2", "GLB_2", "WG_2", "LCL_2", "WI_2", "PRV_2"
        ],
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
    
    Constraints are based on the validation logic from gaussian.py:
    - Work item sizes must not exceed hardware limits
    - Work item dimensions must be different
    - Work group dimensions must be different
    - Divisibility constraints for GLB, WG, LCL, WI, PRV decomposition
    
    Args:
        config: Dictionary with parameter values
        max_wi_size: Tuple of max work item sizes per dimension
        max_wg_size: Max work group size
    
    Returns:
        True if configuration is valid, False otherwise
    """
    cfg = config

    # Work item size constraints
    if cfg['WI_1'] > max_wi_size[cfg['WI_1_OCL_DIM']]:
        return False
    if cfg['WI_2'] > max_wi_size[cfg['WI_2_OCL_DIM']]:
        return False
    if cfg['WI_1'] * cfg['WI_2'] > max_wg_size:
        return False

    # Dimension constraints
    if cfg['WG_1_OCL_DIM'] == cfg['WG_2_OCL_DIM']:
        return False
    if cfg['WI_1_OCL_DIM'] == cfg['WI_2_OCL_DIM']:
        return False

    # Dimension 1 divisibility constraints
    if cfg['INPUT_SIZE_1'] % cfg['GLB_1'] != 0:
        return False
    if (cfg['INPUT_SIZE_1'] / cfg['GLB_1']) % cfg['WG_1'] != 0:
        return False
    if (cfg['INPUT_SIZE_1'] / cfg['GLB_1'] / cfg['WG_1']) % cfg['LCL_1'] != 0:
        return False
    if (cfg['INPUT_SIZE_1'] / cfg['GLB_1'] / cfg['WG_1'] / cfg['LCL_1']) % cfg['WI_1'] != 0:
        return False
    if cfg['INPUT_SIZE_1'] / cfg['GLB_1'] / cfg['WG_1'] / cfg['LCL_1'] / cfg['WI_1'] != cfg['PRV_1']:
        return False

    # Dimension 2 divisibility constraints
    if cfg['INPUT_SIZE_2'] % cfg['GLB_2'] != 0:
        return False
    if (cfg['INPUT_SIZE_2'] / cfg['GLB_2']) % cfg['WG_2'] != 0:
        return False
    if (cfg['INPUT_SIZE_2'] / cfg['GLB_2'] / cfg['WG_2']) % cfg['LCL_2'] != 0:
        return False
    if (cfg['INPUT_SIZE_2'] / cfg['GLB_2'] / cfg['WG_2'] / cfg['LCL_2']) % cfg['WI_2'] != 0:
        return False
    if cfg['INPUT_SIZE_2'] / cfg['GLB_2'] / cfg['WG_2'] / cfg['LCL_2'] / cfg['WI_2'] != cfg['PRV_2']:
        return False

    return True


# =============================================================================
# KERNEL GENERATION
# =============================================================================

def get_kernel_template_path() -> str:
    """Get the path to the Gaussian kernel template.
    
    Returns:
        Absolute path to gaussian_static_1.cl
    """
    script_dir = Path(__file__).parent
    kernel_dir = script_dir.parent / "kernels"
    template_path = kernel_dir / "kernel-template" / "gaussian_static_1.cl"
    return str(template_path.absolute())


def generate_tuned_kernel(
    config: Dict,
    output_path: str,
    template_path: str | None = None,
    device_type: str = "cpu"
) -> bool:
    """Generate a tuned kernel by feeding parameters to the compiler.
    
    This function creates a tuned kernel instance by compiling the kernel template
    with the given configuration parameters as compiler flags. Outputs complete
    .cl file with template.
    
    Args:
        config: Dictionary with parameter names and values
        output_path: Where to write the generated kernel
        template_path: Path to kernel template (defaults to gaussian_static_1.cl)
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

    # Add configuration parameters as preprocessor directives at the top
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
    device_type: str = "cpu",
    input_size_h: int = None,
    input_size_w: int = None
) -> bool:
    """Save tuning parameters and input sizes as JSON file.
    
    Outputs a .json file containing the tuning parameters and input dimensions.
    Used for exhaustive search to generate configuration files.
    
    Args:
        config: Dictionary with parameter names and values
        output_path: Where to write the configuration file (.json extension recommended)
        device_type: Device type ("cpu" or "gpu") - used for validation
        input_size_h: Height input dimension (optional, will use INPUT_SIZE_1 from config if not provided)
        input_size_w: Width input dimension (optional, will use INPUT_SIZE_2 from config if not provided)
    
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

    # Create JSON configuration with input sizes and tuning parameters
    json_config = {}
    
    # Add input sizes
    if input_size_h is not None:
        json_config['input_size_h'] = input_size_h
    elif 'INPUT_SIZE_1' in config:
        json_config['input_size_h'] = config['INPUT_SIZE_1']
    
    if input_size_w is not None:
        json_config['input_size_w'] = input_size_w
    elif 'INPUT_SIZE_2' in config:
        json_config['input_size_w'] = config['INPUT_SIZE_2']
    
    # Add tuning parameters (convert keys to lowercase with underscores)
    for k, v in config.items():
        json_config[k.lower()] = v

    # Write output as JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(json_config, f, indent=2)
        return True
    except IOError as e:
        print(f"Error writing configuration: {e}", file=sys.stderr)
        return False


# =============================================================================
# CONSTRAINT-AWARE CONFIGURATION GENERATION
# =============================================================================

def get_valid_dim1_factors(
    INPUT_SIZE_1: int,
    max_wi_size: Tuple[int, int, int],
    max_wg_size: int,
) -> Dict[str, list]:
    """Generate all valid dimension-1 factor combinations respecting divisibility.
    
    For a decomposition chain: INPUT_SIZE_1 = GLB_1 * WG_1 * LCL_1 * WI_1 * PRV_1,
    generate factors that satisfy:
    - Each factor divides the result of prior factors
    - WI_1 ≤ max_wi_size[1] and WI_1 * WI_2 ≤ max_wg_size
    - PRV_1 is determined by the rest
    
    Args:
        INPUT_SIZE_1: Input size for dimension 1
        max_wi_size: Max work item sizes
        max_wg_size: Max work group size
    
    Returns:
        Dict mapping parameter names to lists of valid values
    """
    result = {
        'GLB_1': [],
        'WG_1': [],
        'LCL_1': [],
        'WI_1': [],
        'PRV_1': [],
    }
    
    # Collect all valid (GLB_1, WG_1, LCL_1, WI_1, PRV_1) tuples
    valid_tuples = []
    
    # Divisors of INPUT_SIZE_1
    glb_1_options = [i for i in range(1, INPUT_SIZE_1 + 1) if INPUT_SIZE_1 % i == 0]
    
    for glb_1 in glb_1_options:
        rem1 = INPUT_SIZE_1 // glb_1
        wg_1_options = [i for i in range(1, rem1 + 1) if rem1 % i == 0]
        
        for wg_1 in wg_1_options:
            rem2 = rem1 // wg_1
            lcl_1_options = [i for i in range(1, rem2 + 1) if rem2 % i == 0]
            
            for lcl_1 in lcl_1_options:
                rem3 = rem2 // lcl_1
                # WI_1 must respect hardware constraints
                wi_1_options = [
                    i for i in range(1, rem3 + 1)
                    if rem3 % i == 0 and i <= max_wi_size[1]
                ]
                
                for wi_1 in wi_1_options:
                    prv_1 = rem3 // wi_1
                    valid_tuples.append((glb_1, wg_1, lcl_1, wi_1, prv_1))
    
    # Flatten into separate lists (each tuple element maps to a unique index)
    if valid_tuples:
        for glb, wg, lcl, wi, prv in valid_tuples:
            result['GLB_1'].append(glb)
            result['WG_1'].append(wg)
            result['LCL_1'].append(lcl)
            result['WI_1'].append(wi)
            result['PRV_1'].append(prv)
    
    return result


def get_valid_dim2_factors(
    INPUT_SIZE_2: int,
    wi_1_value: int,
    max_wi_size: Tuple[int, int, int],
    max_wg_size: int,
) -> Dict[str, list]:
    """Generate all valid dimension-2 factors respecting cross-dim constraints.
    
    Dimension 2 must satisfy: WI_2 ≤ max_wi_size[2] and WI_1 * WI_2 ≤ max_wg_size.
    
    Args:
        INPUT_SIZE_2: Input size for dimension 2
        wi_1_value: Already-chosen WI_1 value (constraints WI_2)
        max_wi_size: Max work item sizes
        max_wg_size: Max work group size
    
    Returns:
        Dict mapping parameter names to lists of valid tuples
    """
    result = {
        'GLB_2': [],
        'WG_2': [],
        'LCL_2': [],
        'WI_2': [],
        'PRV_2': [],
    }
    
    valid_tuples = []
    glb_2_options = [i for i in range(1, INPUT_SIZE_2 + 1) if INPUT_SIZE_2 % i == 0]
    
    for glb_2 in glb_2_options:
        rem1 = INPUT_SIZE_2 // glb_2
        wg_2_options = [i for i in range(1, rem1 + 1) if rem1 % i == 0]
        
        for wg_2 in wg_2_options:
            rem2 = rem1 // wg_2
            lcl_2_options = [i for i in range(1, rem2 + 1) if rem2 % i == 0]
            
            for lcl_2 in lcl_2_options:
                rem3 = rem2 // lcl_2
                # WI_2 must respect hardware constraints AND cross-dimension constraint
                max_wi_2 = min(max_wi_size[2], max_wg_size // wi_1_value)
                wi_2_options = [
                    i for i in range(1, rem3 + 1)
                    if rem3 % i == 0 and i <= max_wi_2
                ]
                
                for wi_2 in wi_2_options:
                    prv_2 = rem3 // wi_2
                    valid_tuples.append((glb_2, wg_2, lcl_2, wi_2, prv_2))
    
    if valid_tuples:
        for glb, wg, lcl, wi, prv in valid_tuples:
            result['GLB_2'].append(glb)
            result['WG_2'].append(wg)
            result['LCL_2'].append(lcl)
            result['WI_2'].append(wi)
            result['PRV_2'].append(prv)
    
    return result


# =============================================================================
# EXHAUSTIVE CONFIGURATION ITERATION
# =============================================================================

def exhaustive_iterate_configurations(
    H: int,
    W: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024
) -> Iterator[Dict]:
    """Exhaustively iterate through all valid configurations.
    
    Generates only valid combinations by enforcing constraints at each parameter level.
    Skips invalid branches early to avoid combinatorial explosion.
    
    Args:
        H: Height dimension
        W: Width dimension
        max_wi_size: Max work item sizes per dimension (CPU or GPU)
        max_wg_size: Max work group size
    
    Yields:
        Valid configuration dictionaries
    """
    # Fixed parameters (based on example)
    fixed_params = {
        'G_CB_RES_DEST_LEVEL': 2,
        'L_CB_RES_DEST_LEVEL': 0,
        'P_CB_RES_DEST_LEVEL': 0,
        'WG_1_OCL_DIM': 1,
        'WG_2_OCL_DIM': 0,
        'WI_1_OCL_DIM': 1,
        'WI_2_OCL_DIM': 0,
        'INPUT_SIZE_1': H,
        'INPUT_SIZE_2': W,
    }

    # Variable parameters with their ranges
    cache_flags = {
        'IMAGES_CACHE_LCL': [0, 1],
        'IMAGES_CACHE_PRV': [0, 1],
        'FILTER_CACHE_LCL': [0, 1],
        'FILTER_CACHE_PRV': [0, 1],
        'OUT_CACHE_PRV': [0, 1],
    }

    cache_flag_names = list(cache_flags.keys())
    cache_flag_values = [cache_flags[name] for name in cache_flag_names]

    # Pre-compute valid dimension 1 factors
    valid_dim1 = get_valid_dim1_factors(H, max_wi_size, max_wg_size)

    # Iterate through all combinations with constraint enforcement
    for cache_combo in product(*cache_flag_values):
        cache_dict = dict(zip(cache_flag_names, cache_combo))

        # Build valid dim1 tuples: each index i has one valid tuple
        for dim1_idx in range(len(valid_dim1['GLB_1'])):
            dim1_dict = {
                'GLB_1': valid_dim1['GLB_1'][dim1_idx],
                'WG_1': valid_dim1['WG_1'][dim1_idx],
                'LCL_1': valid_dim1['LCL_1'][dim1_idx],
                'WI_1': valid_dim1['WI_1'][dim1_idx],
                'PRV_1': valid_dim1['PRV_1'][dim1_idx],
            }

            # Compute valid dimension 2 factors given WI_1
            valid_dim2 = get_valid_dim2_factors(W, dim1_dict['WI_1'], max_wi_size, max_wg_size)

            # Build valid dim2 tuples
            for dim2_idx in range(len(valid_dim2['GLB_2'])):
                dim2_dict = {
                    'GLB_2': valid_dim2['GLB_2'][dim2_idx],
                    'WG_2': valid_dim2['WG_2'][dim2_idx],
                    'LCL_2': valid_dim2['LCL_2'][dim2_idx],
                    'WI_2': valid_dim2['WI_2'][dim2_idx],
                    'PRV_2': valid_dim2['PRV_2'][dim2_idx],
                }

                # Combine all parameters—guaranteed valid
                config = {**fixed_params, **cache_dict, **dim1_dict, **dim2_dict}
                if is_configuration_valid(config, max_wi_size, max_wg_size):
                    yield config



def count_valid_configurations(
    H: int,
    W: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024
) -> int:
    """Count the total number of valid configurations without storing them.
    
    Args:
        H: Height dimension
        W: Width dimension
        max_wi_size: Max work item sizes per dimension
        max_wg_size: Max work group size
    
    Returns:
        Count of valid configurations
    """
    count = 0
    for _ in exhaustive_iterate_configurations(H, W, max_wi_size, max_wg_size):
        count += 1
    return count


def random_sample_configurations(
    H: int,
    W: int,
    num_samples: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024,
    seed: int | None = None,
    attempts_per_sample: int = 1000,
) -> Iterator[Dict]:
    """Randomly *generate* valid configurations without full enumeration.

    Uses constraint-aware random generation: sample cache flags, then uniformly
    sample from pre-computed valid divisor chains for each dimension, ensuring
    all generated configs are valid.

    Args:
        H: Height dimension
        W: Width dimension
        num_samples: Number of configurations to sample
        max_wi_size: Max work item sizes per dimension
        max_wg_size: Max work group size
        seed: Random seed for reproducibility
        attempts_per_sample: Max attempts per desired sample (fallback)

    Yields:
        Random valid configurations (distinct)
    """
    if seed is not None:
        random.seed(seed)

    fixed_params = {
        'G_CB_RES_DEST_LEVEL': 2,
        'L_CB_RES_DEST_LEVEL': 0,
        'P_CB_RES_DEST_LEVEL': 0,
        'WG_1_OCL_DIM': 1,
        'WG_2_OCL_DIM': 0,
        'WI_1_OCL_DIM': 1,
        'WI_2_OCL_DIM': 0,
        'INPUT_SIZE_1': H,
        'INPUT_SIZE_2': W,
    }

    cache_flags = {
        'IMAGES_CACHE_LCL': [0, 1],
        'IMAGES_CACHE_PRV': [0, 1],
        'FILTER_CACHE_LCL': [0, 1],
        'FILTER_CACHE_PRV': [0, 1],
        'OUT_CACHE_PRV': [0, 1],
    }

    # Pre-compute valid dimension 1 and 2 factor chains
    valid_dim1 = get_valid_dim1_factors(H, max_wi_size, max_wg_size)
    num_dim1_options = len(valid_dim1['GLB_1'])

    if num_dim1_options == 0:
        print(f"Warning: No valid dimension 1 configurations for H={H}", file=sys.stderr)
        return

    cache_flag_names = list(cache_flags.keys())
    cache_flag_values = [cache_flags[name] for name in cache_flag_names]

    seen = set()
    target = num_samples

    # Generate samples by random selection from constraint-valid space
    for attempt in range(max(num_samples * attempts_per_sample, attempts_per_sample)):
        if target <= 0:
            break

        # Random cache flags
        cache_combo = tuple(random.choice(v) for v in cache_flag_values)
        cache_dict = dict(zip(cache_flag_names, cache_combo))

        # Random dimension 1 choice from valid options
        dim1_idx = random.randint(0, num_dim1_options - 1)
        dim1_dict = {
            'GLB_1': valid_dim1['GLB_1'][dim1_idx],
            'WG_1': valid_dim1['WG_1'][dim1_idx],
            'LCL_1': valid_dim1['LCL_1'][dim1_idx],
            'WI_1': valid_dim1['WI_1'][dim1_idx],
            'PRV_1': valid_dim1['PRV_1'][dim1_idx],
        }

        # Compute valid dimension 2 options given WI_1
        valid_dim2 = get_valid_dim2_factors(W, dim1_dict['WI_1'], max_wi_size, max_wg_size)
        num_dim2_options = len(valid_dim2['GLB_2'])

        if num_dim2_options == 0:
            continue  # Skip invalid WI_1 choice

        # Random dimension 2 choice from valid options
        dim2_idx = random.randint(0, num_dim2_options - 1)
        dim2_dict = {
            'GLB_2': valid_dim2['GLB_2'][dim2_idx],
            'WG_2': valid_dim2['WG_2'][dim2_idx],
            'LCL_2': valid_dim2['LCL_2'][dim2_idx],
            'WI_2': valid_dim2['WI_2'][dim2_idx],
            'PRV_2': valid_dim2['PRV_2'][dim2_idx],
        }

        config = {**fixed_params, **cache_dict, **dim1_dict, **dim2_dict}

        # Fast de-duplication check
        key = tuple(config.items())
        if key in seen:
            continue

        seen.add(key)
        target -= 1
        yield config

    if target > 0:
        print(
            f"Warning: Requested {num_samples} configs but only {num_samples - target} "
            f"could be generated after {attempts_per_sample} attempts.",
            file=sys.stderr,
        )


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
    H, W = args.input_size[0], args.input_size[1]
    manipulator = get_parameters_definition(H, W)

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
    # Load configuration
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
    H, W = args.input_size[0], args.input_size[1]

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
        # Just count valid configurations
        count = count_valid_configurations(H, W, max_wi_size, max_wg_size)
        print(f"Total valid configurations: {count}")
        return 0

    # Choose iteration strategy
    if args.random:
        # Random sampling mode (on-the-fly rejection sampling)
        num_samples = args.random
        seed = args.seed
        attempts_per_sample = args.random_attempts_per_sample
        config_iter = random_sample_configurations(
            H, W, num_samples, max_wi_size, max_wg_size, seed, attempts_per_sample
        )
        if args.verbose:
            print(
                f"Sampling {num_samples} random configurations (seed={seed}, "
                f"attempts/sample={attempts_per_sample})"
            )
    else:
        # Sequential exhaustive mode
        config_iter = exhaustive_iterate_configurations(H, W, max_wi_size, max_wg_size)
        if args.verbose and args.max:
            print(f"Generating first {args.max} configurations sequentially")

    # Generate kernels
    generated = 0
    invalid = 0
    for i, config in enumerate(config_iter):
        if args.max and generated >= args.max:
            break

        # Create output filename with .json extension
        output_file = os.path.join(output_dir, f"gaussian_{H}x{W}_{i:06d}.json")

        # Save tuning parameters as JSON with input sizes
        if save_tuning_parameters_only(config, output_file, device_type=args.device_type,
                                      input_size_h=H, input_size_w=W):
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
        description="Generate tuned Gaussian kernels with different parameter configurations"
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
    parser_size.add_argument('input_size', type=int, nargs=2, metavar=('H', 'W'),
                             help='Input dimensions (H W)')
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
    parser_exh.add_argument('input_size', type=int, nargs=2, metavar=('H', 'W'),
                            help='Input dimensions (H W)')
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
