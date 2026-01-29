"""Generate tuned versions of Relation (RL) kernels using OpenTuner.

This script provides:
1. Parameter definition separated from compilation logic
2. Tuned kernel generation by feeding parameters to the compiler
3. Exhaustive configuration iteration through all possible valid configurations
4. CLI for fast testing and batch generation

Usage:
    python -m scripts.generate_rl --help
    python -m scripts.generate_rl list-params
    python -m scripts.generate_rl generate --config config.json --output kernel.cl
    python -m scripts.generate_rl exhaustive --input-size 1000000 1000000 --output-dir ./kernels/
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

def get_parameters_definition(M: int, N: int) -> ConfigurationManipulator:
    """Define the OpenTuner parameter search space for RL kernel tuning.
    
    This function encapsulates all tuning parameters and their constraints
    based on the OpenTuner example in rl.cpp.
    
    Args:
        M: M dimension size
        N: N dimension size
    
    Returns:
        ConfigurationManipulator with all parameters and ranges defined
    """
    manipulator = ConfigurationManipulator()

    # Cache parameters - control where data is cached
    manipulator.add_parameter(IntegerParameter('CACHE_L_CB', 0, 1))
    manipulator.add_parameter(IntegerParameter('CACHE_P_CB', 0, 1))
    manipulator.add_parameter(IntegerParameter('G_CB_RES_DEST_LEVEL', 2, 2))
    manipulator.add_parameter(IntegerParameter('L_CB_RES_DEST_LEVEL', 0, 2))
    manipulator.add_parameter(IntegerParameter('P_CB_RES_DEST_LEVEL', 0, 2))

    # OCL dimensions - 2D work space
    manipulator.add_parameter(IntegerParameter('OCL_DIM_L_1', 0, 1))
    manipulator.add_parameter(IntegerParameter('OCL_DIM_R_1', 0, 1))

    # L_1 (M) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_L_1', M, M))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_L_1', 1, M))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_L_1', 1, M))
    manipulator.add_parameter(IntegerParameter('NUM_WG_L_1', 1, M))
    manipulator.add_parameter(IntegerParameter('NUM_WI_L_1', 1, M))

    # R_1 (N) dimension parameters
    manipulator.add_parameter(IntegerParameter('INPUT_SIZE_R_1', N, N))
    manipulator.add_parameter(IntegerParameter('L_CB_SIZE_R_1', 1, N))
    manipulator.add_parameter(IntegerParameter('P_CB_SIZE_R_1', 1, N))
    manipulator.add_parameter(IntegerParameter('NUM_WG_R_1', 1, N))
    manipulator.add_parameter(IntegerParameter('NUM_WI_R_1', 1, N))

    # Fixed parameters for write-back and reduction
    manipulator.add_parameter(IntegerParameter('L_REDUCTION', 1, 1))
    manipulator.add_parameter(IntegerParameter('P_WRITE_BACK', 0, 0))
    manipulator.add_parameter(IntegerParameter('L_WRITE_BACK', 1, 1))

    return manipulator


def get_parameters_info() -> Dict:
    """Return a dictionary with metadata about all tuning parameters.
    
    Returns:
        Dict with parameter names, ranges, and descriptions
    """
    return {
        "cache_parameters": {
            "CACHE_L_CB": "Cache at local level for L matrices",
            "CACHE_P_CB": "Cache at private level",
            "G_CB_RES_DEST_LEVEL": "Global cache resolution destination level",
            "L_CB_RES_DEST_LEVEL": "Local cache resolution destination level",
            "P_CB_RES_DEST_LEVEL": "Private cache resolution destination level",
        },
        "ocl_dimensions": {
            "OCL_DIM_L_1": "OpenCL dimension for L_1 (M) in {0, 1}",
            "OCL_DIM_R_1": "OpenCL dimension for R_1 (N) in {0, 1}",
        },
        "dimension_l1_parameters": [
            "INPUT_SIZE_L_1", "L_CB_SIZE_L_1", "P_CB_SIZE_L_1", "NUM_WG_L_1", "NUM_WI_L_1"
        ],
        "dimension_r1_parameters": [
            "INPUT_SIZE_R_1", "L_CB_SIZE_R_1", "P_CB_SIZE_R_1", "NUM_WG_R_1", "NUM_WI_R_1"
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
    
    Constraints are based on the validation logic from rl.cpp:
    - Work item sizes must not exceed hardware limits
    - Work item dimensions must be different
    - Divisibility constraints for L_CB_SIZE, P_CB_SIZE, NUM_WG, NUM_WI decomposition
    
    Args:
        config: Dictionary with parameter values
        max_wi_size: Tuple of max work item sizes per dimension
        max_wg_size: Max work group size
    
    Returns:
        True if configuration is valid, False otherwise
    """
    cfg = config

    # Work item size constraints
    if cfg['NUM_WI_L_1'] > max_wi_size[cfg['OCL_DIM_L_1']]:
        return False
    if cfg['NUM_WI_R_1'] > max_wi_size[cfg['OCL_DIM_R_1']]:
        return False
    if cfg['NUM_WI_L_1'] * cfg['NUM_WI_R_1'] > max_wg_size:
        return False

    # Dimension constraints - must be different
    if cfg['OCL_DIM_L_1'] == cfg['OCL_DIM_R_1']:
        return False

    # L_1 divisibility constraints: INPUT_SIZE = L_CB_SIZE * NUM_WG * NUM_WI * P_CB_SIZE
    if cfg['INPUT_SIZE_L_1'] % cfg['L_CB_SIZE_L_1'] != 0:
        return False
    rem = cfg['INPUT_SIZE_L_1'] // cfg['L_CB_SIZE_L_1']
    if rem % cfg['NUM_WG_L_1'] != 0:
        return False
    rem = rem // cfg['NUM_WG_L_1']
    if rem % cfg['NUM_WI_L_1'] != 0:
        return False
    rem = rem // cfg['NUM_WI_L_1']
    if rem != cfg['P_CB_SIZE_L_1']:
        return False

    # R_1 divisibility constraints
    if cfg['INPUT_SIZE_R_1'] % cfg['L_CB_SIZE_R_1'] != 0:
        return False
    rem = cfg['INPUT_SIZE_R_1'] // cfg['L_CB_SIZE_R_1']
    if rem % cfg['NUM_WG_R_1'] != 0:
        return False
    rem = rem // cfg['NUM_WG_R_1']
    if rem % cfg['NUM_WI_R_1'] != 0:
        return False
    rem = rem // cfg['NUM_WI_R_1']
    if rem != cfg['P_CB_SIZE_R_1']:
        return False

    return True


# =============================================================================
# KERNEL GENERATION
# =============================================================================

def get_kernel_template_path() -> str:
    """Get the path to the RL kernel template.
    
    Returns:
        Absolute path to rl_1.cl
    """
    script_dir = Path(__file__).parent
    kernel_dir = script_dir.parent / "kernels"
    template_path = kernel_dir / "kernel-template" / "rl_1.cl"
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
        template_path: Path to kernel template (defaults to rl_1.cl)
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
        print(f"Error: Invalid configuration", file=sys.stderr)
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
    device_type: str = "cpu"
) -> bool:
    """Save only the tuning parameter definitions without the template.
    
    Outputs a .cl file containing only the #define directives for tuning parameters.
    Used for exhaustive search to generate compact parameter definition files.
    
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
        print(f"Error: Invalid configuration", file=sys.stderr)
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
    max_wi_size_for_dim: int,
    max_wg_size: int,
    other_wi_product: int = 1,
) -> Dict[str, list]:
    """Generate all valid dimension factor combinations respecting divisibility.
    
    For a decomposition: INPUT_SIZE = L_CB_SIZE * NUM_WG * NUM_WI * P_CB_SIZE,
    generate factors that satisfy all constraints.
    
    Args:
        INPUT_SIZE: Input size for this dimension
        max_wi_size_for_dim: Max work item size for this dimension
        max_wg_size: Max work group size (considering other dimensions)
        other_wi_product: Product of NUM_WI values from other dimensions
    
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
    
    # Divisors of INPUT_SIZE for L_CB_SIZE
    lcb_options = [i for i in range(1, INPUT_SIZE + 1) if INPUT_SIZE % i == 0]
    
    for lcb in lcb_options:
        rem1 = INPUT_SIZE // lcb
        wg_options = [i for i in range(1, rem1 + 1) if rem1 % i == 0]
        
        for wg in wg_options:
            rem2 = rem1 // wg
            wi_options = [
                i for i in range(1, rem2 + 1)
                if rem2 % i == 0 
                and i <= max_wi_size_for_dim
                and i * other_wi_product <= max_wg_size
            ]
            
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
    M: int,
    N: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024
) -> Iterator[Dict]:
    """Exhaustively iterate through all valid configurations.
    
    Generates only valid combinations by enforcing constraints at each parameter level.
    Skips invalid branches early to avoid combinatorial explosion.
    
    Args:
        M: M dimension size
        N: N dimension size
        max_wi_size: Max work item sizes per dimension (CPU or GPU)
        max_wg_size: Max work group size
    
    Yields:
        Valid configuration dictionaries
    """
    # Fixed parameters
    fixed_params = {
        'G_CB_RES_DEST_LEVEL': 2,
        'L_REDUCTION': 1,
        'P_WRITE_BACK': 0,
        'L_WRITE_BACK': 1,
        'INPUT_SIZE_L_1': M,
        'INPUT_SIZE_R_1': N,
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

    # OCL dimension choices - must be different
    ocl_dim_choices = [(0, 1), (1, 0)]

    # Pre-compute valid dimension factors
    valid_l1 = get_valid_dimension_factors(M, max_wi_size[0], max_wg_size)
    valid_r1 = get_valid_dimension_factors(N, max_wi_size[1], max_wg_size)

    cache_flag_names = list(cache_params.keys())
    cache_flag_values = [cache_params[name] for name in cache_flag_names]
    
    cb_res_names = list(cb_res_levels.keys())
    cb_res_values = [cb_res_levels[name] for name in cb_res_names]

    # Iterate through all combinations with constraint enforcement
    for cache_combo in product(*cache_flag_values):
        cache_dict = dict(zip(cache_flag_names, cache_combo))
        
        for cb_res_combo in product(*cb_res_values):
            # Check hierarchy constraint: P_CB_RES_DEST_LEVEL <= L_CB_RES_DEST_LEVEL
            if cb_res_combo[1] > cb_res_combo[0]:
                continue
            
            cb_res_dict = dict(zip(cb_res_names, cb_res_combo))

            # Iterate dimension combinations
            for ocl_dims in ocl_dim_choices:
                ocl_dict = {
                    'OCL_DIM_L_1': ocl_dims[0],
                    'OCL_DIM_R_1': ocl_dims[1],
                }

                # Build valid L_1 tuples
                for l1_idx in range(len(valid_l1['L_CB_SIZE'])):
                    l1_dict = {
                        'L_CB_SIZE_L_1': valid_l1['L_CB_SIZE'][l1_idx],
                        'NUM_WG_L_1': valid_l1['NUM_WG'][l1_idx],
                        'NUM_WI_L_1': valid_l1['NUM_WI'][l1_idx],
                        'P_CB_SIZE_L_1': valid_l1['P_CB_SIZE'][l1_idx],
                    }

                    # Build valid R_1 tuples with cross-dimension constraint
                    wi_product = l1_dict['NUM_WI_L_1']
                    valid_r1_filtered = get_valid_dimension_factors(
                        N, max_wi_size[1], max_wg_size, wi_product
                    )

                    for r1_idx in range(len(valid_r1_filtered['L_CB_SIZE'])):
                        r1_dict = {
                            'L_CB_SIZE_R_1': valid_r1_filtered['L_CB_SIZE'][r1_idx],
                            'NUM_WG_R_1': valid_r1_filtered['NUM_WG'][r1_idx],
                            'NUM_WI_R_1': valid_r1_filtered['NUM_WI'][r1_idx],
                            'P_CB_SIZE_R_1': valid_r1_filtered['P_CB_SIZE'][r1_idx],
                        }

                        # Combine all parameters
                        config = {
                            **fixed_params,
                            **cache_dict,
                            **cb_res_dict,
                            **ocl_dict,
                            **l1_dict,
                            **r1_dict,
                        }
                        
                        if is_configuration_valid(config, max_wi_size, max_wg_size):
                            yield config


def count_valid_configurations(
    M: int,
    N: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024
) -> int:
    """Count the total number of valid configurations without storing them.
    
    Args:
        M: M dimension size
        N: N dimension size
        max_wi_size: Max work item sizes per dimension
        max_wg_size: Max work group size
    
    Returns:
        Count of valid configurations
    """
    count = 0
    for _ in exhaustive_iterate_configurations(M, N, max_wi_size, max_wg_size):
        count += 1
    return count


def random_sample_configurations(
    M: int,
    N: int,
    num_samples: int,
    max_wi_size: Tuple[int, int, int] = (1024, 1024, 64),
    max_wg_size: int = 1024,
    seed: int | None = None,
    attempts_per_sample: int = 1000,
) -> Iterator[Dict]:
    """Randomly generate valid configurations without full enumeration.

    Uses constraint-aware random generation: sample cache flags, then uniformly
    sample from pre-computed valid divisor chains for each dimension, ensuring
    all generated configs are valid.

    Args:
        M: M dimension size
        N: N dimension size
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
        'L_REDUCTION': 1,
        'P_WRITE_BACK': 0,
        'L_WRITE_BACK': 1,
        'INPUT_SIZE_L_1': M,
        'INPUT_SIZE_R_1': N,
    }

    cache_params = {
        'CACHE_L_CB': [0, 1],
        'CACHE_P_CB': [0, 1],
    }

    cb_res_levels = {
        'L_CB_RES_DEST_LEVEL': [0, 1, 2],
        'P_CB_RES_DEST_LEVEL': [0, 1, 2],
    }

    ocl_dim_choices = [(0, 1), (1, 0)]

    # Pre-compute valid dimension factors
    valid_l1 = get_valid_dimension_factors(M, max_wi_size[0], max_wg_size)

    cache_flag_names = list(cache_params.keys())
    cache_flag_values = [cache_params[name] for name in cache_flag_names]
    
    cb_res_names = list(cb_res_levels.keys())
    cb_res_values = [cb_res_levels[name] for name in cb_res_names]

    seen = set()
    generated = 0
    valid = 0

    num_l1_options = len(valid_l1['L_CB_SIZE'])

    if num_l1_options == 0:
        print(f"Warning: No valid dimension configurations for M={M}", file=sys.stderr)
        return

    # Generate samples by random selection until N valid kernels are produced
    while valid < num_samples:

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
        ocl_dims = random.choice(ocl_dim_choices)
        ocl_dict = {
            'OCL_DIM_L_1': ocl_dims[0],
            'OCL_DIM_R_1': ocl_dims[1],
        }

        # Random dimension choices
        l1_idx = random.randint(0, num_l1_options - 1)
        l1_dict = {
            'L_CB_SIZE_L_1': valid_l1['L_CB_SIZE'][l1_idx],
            'NUM_WG_L_1': valid_l1['NUM_WG'][l1_idx],
            'NUM_WI_L_1': valid_l1['NUM_WI'][l1_idx],
            'P_CB_SIZE_L_1': valid_l1['P_CB_SIZE'][l1_idx],
        }

        # R_1 with cross-dimension constraint
        wi_product = l1_dict['NUM_WI_L_1']
        valid_r1_filtered = get_valid_dimension_factors(
            N, max_wi_size[1], max_wg_size, wi_product
        )
        num_r1_options = len(valid_r1_filtered['L_CB_SIZE'])

        if num_r1_options == 0:
            continue

        r1_idx = random.randint(0, num_r1_options - 1)
        r1_dict = {
            'L_CB_SIZE_R_1': valid_r1_filtered['L_CB_SIZE'][r1_idx],
            'NUM_WG_R_1': valid_r1_filtered['NUM_WG'][r1_idx],
            'NUM_WI_R_1': valid_r1_filtered['NUM_WI'][r1_idx],
            'P_CB_SIZE_R_1': valid_r1_filtered['P_CB_SIZE'][r1_idx],
        }

        config = {
            **fixed_params,
            **cache_dict,
            **cb_res_dict,
            **ocl_dict,
            **l1_dict,
            **r1_dict,
        }

        # Fast de-duplication check
        key = tuple(config.items())
        if key in seen:
            continue

        seen.add(key)
        generated += 1
        if is_configuration_valid(config, max_wi_size, max_wg_size):
            valid += 1
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
    M, N = args.input_size[0], args.input_size[1]
    manipulator = get_parameters_definition(M, N)

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
    M, N = args.input_size[0], args.input_size[1]

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
        count = count_valid_configurations(M, N, max_wi_size, max_wg_size)
        print(f"Total valid configurations: {count}")
        return 0

    # Choose iteration strategy
    if args.random:
        # Random sampling mode
        num_samples = args.random
        seed = args.seed
        attempts_per_sample = args.random_attempts_per_sample
        config_iter = random_sample_configurations(
            M, N, num_samples, max_wi_size, max_wg_size, seed, attempts_per_sample
        )
        if args.verbose:
            print(
                f"Sampling {num_samples} random configurations (seed={seed}, "
                f"attempts/sample={attempts_per_sample})"
            )
    else:
        # Sequential exhaustive mode
        config_iter = exhaustive_iterate_configurations(M, N, max_wi_size, max_wg_size)
        if args.verbose and args.max:
            print(f"Generating first {args.max} configurations sequentially")

    # Generate kernels
    generated = 0
    invalid = 0
    for i, config in enumerate(config_iter):
        if args.max and generated >= args.max:
            break

        # Create output filename
        output_file = os.path.join(output_dir, f"rl_{M}x{N}_{i:06d}.cl")

        # Save tuning parameters only (no json, no template)
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
        description="Generate tuned RL (Relation) kernels with different parameter configurations"
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
    parser_size.add_argument('input_size', type=int, nargs=2, metavar=('M', 'N'),
                             help='Input dimensions (M N)')
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
    parser_exh.add_argument('input_size', type=int, nargs=2, metavar=('M', 'N'),
                            help='Input dimensions (M N)')
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
