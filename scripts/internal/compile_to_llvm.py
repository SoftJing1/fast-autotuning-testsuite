"""Compile OpenCL kernels to standalone LLVM IR.

This script automates the compilation of OpenCL kernel templates with tuning
parameters to produce standalone LLVM IR files ready for executable compilation.

Usage:
    python -m scripts.compile_to_llvm --help
    python -m scripts.internal.compile_to_llvm compile --template kernel-template/gaussian_static_1.cl --params tuning_params/gaussian/gaussian_1024x1024_000000.cl --output output.ll
    python -m scripts.internal.compile_to_llvm batch --template kernel-template/gaussian_static_1.cl --params-dir tuning_params/gaussian --output-dir ./llvmir/ --max 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple


# =============================================================================
# OPENCL COMPILATION UTILITIES
# =============================================================================


def merge_opencl_files(
    template_path: str,
    params_path: str,
    output_path: str | None = None
) -> str:
    """Merge kernel template with tuning parameter definitions.
    
    Combines the parameter definitions from params_path with the kernel
    template to create a complete OpenCL source file.
    
    Args:
        template_path: Path to kernel template (.cl file)
        params_path: Path to tuning parameters definition (.cl file)
        output_path: Optional path to save merged file (returns content if None)
    
    Returns:
        Content of merged file or path if output_path specified
    """
    # Read template
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error reading template: {e}", file=sys.stderr)
        raise

    # Read parameters
    try:
        with open(params_path, 'r') as f:
            params_content = f.read()
    except IOError as e:
        print(f"Error reading parameters: {e}", file=sys.stderr)
        raise

    # Merge: parameters first, then template
    merged_content = params_content + "\n\n" + template_content

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            with open(output_path, 'w') as f:
                f.write(merged_content)
            return output_path
        except IOError as e:
            print(f"Error writing merged file: {e}", file=sys.stderr)
            raise

    return merged_content


def detect_opencl_compiler() -> str:
    """Detect available OpenCL compiler (clang or other).
    
    Returns:
        Path to OpenCL compiler command
    
    Raises:
        RuntimeError if no compiler found
    """
    # Try clang with OpenCL support
    try:
        result = subprocess.run(['clang', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'clang'
    except FileNotFoundError:
        pass

    # Try other common OpenCL compilers
    for compiler in ['clang-14', 'clang-13', 'clang-12', 'gcc']:
        try:
            subprocess.run([compiler, '--version'], capture_output=True, text=True, check=True)
            return compiler
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    raise RuntimeError("No suitable OpenCL compiler found. Install clang with OpenCL support.")


def compile_opencl_to_llvm(
    template_path: str,
    params_path: str,
    output_ll_path: str,
    target_triple: str | None = None,
    data_layout: str | None = None
) -> bool:
    """Compile OpenCL template + params to LLVM IR without merging files.

    Uses clang with `-include <params>` so the tuning parameter definitions are
    pulled in without concatenating files on disk.

    Args:
        template_path: Path to kernel template (.cl)
        params_path: Path to tuning parameter definitions (.cl)
        output_ll_path: Path to output LLVM IR file (.ll)
        target_triple: Optional target triple (e.g., x86_64-unknown-linux-gnu)
        data_layout: Optional data layout specification

    Returns:
        True if compilation succeeded, False otherwise
    """
    os.makedirs(os.path.dirname(output_ll_path) or ".", exist_ok=True)

    try:
        compiler = detect_opencl_compiler()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    # Build clang command for LLVM IR generation
    cmd = [
        compiler,
        '-emit-llvm',
        '-S',
        '-x', 'cl',
        '-cl-std=CL2.0',
        '-include', params_path,
        template_path,
        '-o', output_ll_path
    ]

    if target_triple:
        cmd.extend(['-triple', target_triple])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"Compilation error:\n{result.stderr}", file=sys.stderr)
            return False

        if not os.path.exists(output_ll_path):
            print(f"Error: Output file not created", file=sys.stderr)
            return False

        return True
    except Exception as e:
        print(f"Error running compiler: {e}", file=sys.stderr)
        return False


def compile_kernel_to_standalone_llvm(
    template_path: str,
    params_path: str,
    output_ll_path: str,
    target_triple: str | None = None,
    verbose: bool = False
) -> bool:
    """Complete pipeline: compile template + params to standalone LLVM IR.

    Uses clang `-include <params>` to avoid merging files on disk.

    Args:
        template_path: Path to kernel template (.cl)
        params_path: Path to tuning parameters file (.cl)
        output_ll_path: Output LLVM IR file path
        target_triple: Optional target triple
        verbose: Print progress information

    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(template_path):
            print(f"Error: Template not found: {template_path}", file=sys.stderr)
            return False

        if not os.path.exists(params_path):
            print(f"Error: Parameters file not found: {params_path}", file=sys.stderr)
            return False

        if verbose:
            print(f"Template: {template_path}")
            print(f"Parameters: {params_path}")

        if verbose:
            print(f"Compiling to LLVM IR...")

        success = compile_opencl_to_llvm(template_path, params_path, output_ll_path, target_triple)

        if success and verbose:
            print(f"Output: {output_ll_path}")

        return success

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


# =============================================================================
# CLI
# =============================================================================

def cmd_compile(args) -> int:
    """Compile a single kernel to LLVM IR."""
    success = compile_kernel_to_standalone_llvm(
        args.template,
        args.params,
        args.output,
        target_triple=args.target_triple,
        verbose=args.verbose
    )
    return 0 if success else 1


def cmd_batch(args) -> int:
    """Batch compile multiple kernels to LLVM IR."""
    params_dir = args.params_dir
    output_dir = args.output_dir
    template_path = args.template
    max_count = args.max
    verbose = args.verbose

    if not os.path.isdir(params_dir):
        print(f"Error: Parameters directory not found: {params_dir}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    # Find all parameter files
    params_files = []
    for entry in os.listdir(params_dir):
        if entry.endswith('.cl'):
            params_files.append(os.path.join(params_dir, entry))

    params_files.sort()

    if max_count:
        params_files = params_files[:max_count]

    if not params_files:
        print(f"Error: No .cl files found in {params_dir}", file=sys.stderr)
        return 1

    if verbose:
        print(f"Found {len(params_files)} parameter files")

    compiled = 0
    failed = 0

    for i, params_path in enumerate(params_files):
        params_name = os.path.basename(params_path)
        output_ll_path = os.path.join(output_dir, params_name.replace('.cl', '.ll'))

        if verbose:
            print(f"\n[{i+1}/{len(params_files)}] Compiling {params_name}...")

        success = compile_kernel_to_standalone_llvm(
            template_path,
            params_path,
            output_ll_path,
            target_triple=args.target_triple,
            verbose=verbose
        )

        if success:
            compiled += 1
            if verbose:
                print(f"  ✓ {output_ll_path}")
        else:
            failed += 1
            if verbose:
                print(f"  ✗ Failed")

    print(f"\nSummary: {compiled} compiled, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_list_templates(args) -> int:
    """List available kernel templates (default kernel-template dir)."""
    script_dir = Path(__file__).parent
    kernel_dir = script_dir.parent.parent / "kernels"
    template_dir = kernel_dir / "kernel-template"

    if not template_dir.exists():
        print(f"Error: kernel-template directory not found at {template_dir}", file=sys.stderr)
        return 1

    templates = sorted([f for f in os.listdir(template_dir) if f.endswith('.cl')])

    if not templates:
        print("No templates found")
        return 0

    print(f"Available kernel templates in {template_dir}:")
    for template in templates:
        print(f"  - {template}")

    return 0


def main():
    """Parse arguments and dispatch to appropriate command."""
    parser = argparse.ArgumentParser(
        description="Compile OpenCL kernels to standalone LLVM IR"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # compile command
    parser_compile = subparsers.add_parser(
        'compile',
        help='Compile a single kernel to LLVM IR'
    )
    parser_compile.add_argument('--template', required=True,
                                help='Path to kernel template (.cl)')
    parser_compile.add_argument('--params', required=True,
                                help='Path to tuning parameters file')
    parser_compile.add_argument('--output', required=True,
                                help='Output LLVM IR file path')
    parser_compile.add_argument('--target-triple', default=None,
                                help='Target triple (e.g., x86_64-unknown-linux-gnu)')
    parser_compile.add_argument('--verbose', action='store_true',
                                help='Print progress information')
    parser_compile.set_defaults(func=cmd_compile)

    # batch command
    parser_batch = subparsers.add_parser(
        'batch',
        help='Batch compile kernels from a directory'
    )
    parser_batch.add_argument('--template', required=True,
                              help='Path to kernel template (.cl)')
    parser_batch.add_argument('--params-dir', required=True,
                              help='Directory containing parameter files')
    parser_batch.add_argument('--output-dir', required=True,
                              help='Output directory for LLVM IR files')
    parser_batch.add_argument('--max', type=int, default=None,
                              help='Maximum number of files to compile')
    parser_batch.add_argument('--target-triple', default=None,
                              help='Target triple (e.g., x86_64-unknown-linux-gnu)')
    parser_batch.add_argument('--verbose', action='store_true',
                              help='Print progress information')
    parser_batch.set_defaults(func=cmd_batch)

    # list-templates command
    subparsers.add_parser(
        'list-templates',
        help='List available kernel templates'
    ).set_defaults(func=cmd_list_templates)

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
