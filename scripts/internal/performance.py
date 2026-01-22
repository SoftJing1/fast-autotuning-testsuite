"""
Automate performance measurement for tuned kernels (gaussian, gemm, rl, tc).

This script provides a clean function API and a CLI for measuring the runtime of a tuned kernel instance (in LLVM IR).
It infers the kernel type, generates a host main function, compiles the kernel and host code, and runs the executable multiple times to collect timing statistics.

Usage (CLI):
    python -m scripts.internal.performance --kernel-ll /path/to/tuned_kernel.ll --input-size 1024 --repeat 10

Function API:
    measure_kernel_performance(kernel_ll_path, input_size, repeat=10) -> dict

Inputs:
    - kernel_ll_path: Path to tuned kernel LLVM IR file
    - input_size: Size of input (type depends on kernel)
    - repeat: Number of runs for timing statistics

Outputs:
    - Prints or returns timing statistics (min, max, mean, std)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import statistics
from pathlib import Path
from typing import Dict, Any

# ===============================
# FUNCTION API
# ===============================

def infer_kernel_type(kernel_ll_path: str) -> str:
    """Infer kernel type from function names in LLVM IR."""
    with open(kernel_ll_path, 'r') as f:
        content = f.read()
    # Simple heuristics: look for known kernel entry names
    if 'gaussian_' in content:
        return 'conv'
    elif 'gemm_' in content:
        return 'gemm'
    elif 'rl_' in content:
        return 'rl'
    elif 'tc_' in content:
        return 'tc'
    else:
        raise ValueError("Unknown kernel type in LLVM IR")


def generate_host_code(kernel_type: str, input_size: int) -> str:
    """Generate C host code to call the tuned kernel, allocate buffers, and measure time."""
    # For simplicity, use a generic template per kernel type
    # This can be extended for more complex input shapes
    if kernel_type == 'conv':
        main_code = f'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void gaussian_1(float* input, float* empty, float* output);
int main() {{
    int N = {input_size};
    float* input = (float*)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) input[i] = (float)i;
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    gaussian_1(input, NULL, output);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec-t1.tv_sec)*1000.0 + (t2.tv_nsec-t1.tv_nsec)/1e6;
    printf("%f\\n", elapsed);
    free(input); free(output);
    return 0;
}}
'''
    elif kernel_type == 'gemm':
        main_code = f'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void gemm_1(float* A, float* B, float* C, int N);
int main() {{
    int N = {input_size};
    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));
    float* C = (float*)malloc(N*N*sizeof(float));
    for (int i = 0; i < N*N; ++i) {{ A[i]=1.0f; B[i]=2.0f; }}
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    gemm_1(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec-t1.tv_sec)*1000.0 + (t2.tv_nsec-t1.tv_nsec)/1e6;
    printf("%f\\n", elapsed);
    free(A); free(B); free(C);
    return 0;
}}
'''
    elif kernel_type == 'rl':
        main_code = f'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void rl_1(float* input, float* output, int N);
int main() {{
    int N = {input_size};
    float* input = (float*)malloc(N*sizeof(float));
    float* output = (float*)malloc(N*sizeof(float));
    for (int i = 0; i < N; ++i) input[i] = (float)i;
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    rl_1(input, output, N);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec-t1.tv_sec)*1000.0 + (t2.tv_nsec-t1.tv_nsec)/1e6;
    printf("%f\\n", elapsed);
    free(input); free(output);
    return 0;
}}
'''
    elif kernel_type == 'tc':
        main_code = f'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void tc_abcdef_gebc_dfga_1(float* input, float* output, int N);
int main() {{
    int N = {input_size};
    float* input = (float*)malloc(N*sizeof(float));
    float* output = (float*)malloc(N*sizeof(float));
    for (int i = 0; i < N; ++i) input[i] = (float)i;
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    tc_abcdef_gebc_dfga_1(input, output, N);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed = (t2.tv_sec-t1.tv_sec)*1000.0 + (t2.tv_nsec-t1.tv_nsec)/1e6;
    printf("%f\\n", elapsed);
    free(input); free(output);
    return 0;
}}
'''
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    return main_code


def compile_and_link(kernel_ll_path: str, host_code: str, output_exe: str) -> bool:
    """Compile LLVM IR and host code to executable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ll_path = os.path.join(tmpdir, "kernel.ll")
        host_path = os.path.join(tmpdir, "host.c")
        obj_path = os.path.join(tmpdir, "kernel.o")
        host_obj_path = os.path.join(tmpdir, "host.o")
        # Write files
        with open(ll_path, 'w') as f:
            with open(kernel_ll_path, 'r') as src:
                f.write(src.read())
        with open(host_path, 'w') as f:
            f.write(host_code)
        # Compile kernel LLVM IR to object
        cmd1 = ["clang", "-c", ll_path, "-o", obj_path]
        # Compile host code
        cmd2 = ["clang", "-O2", "-c", host_path, "-o", host_obj_path]
        # Link with OpenCL library
        cmd3 = ["clang", "-lOpenCL", obj_path, host_obj_path, "-o", output_exe]
        for cmd in [cmd1, cmd2, cmd3]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"Running command: {' '.join(cmd)}")
            if result.returncode != 0:
                print(f"Error: {result.stderr}", file=sys.stderr)
                return False
    return True


def run_executable(exe_path: str, repeat: int) -> Dict[str, Any]:
    """Run the executable multiple times and collect timing statistics."""
    times = []
    for _ in range(repeat):
        result = subprocess.run([exe_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Execution error: {result.stderr}", file=sys.stderr)
            continue
        try:
            t = float(result.stdout.strip().split('\n')[-1])
            times.append(t)
        except Exception:
            print(f"Failed to parse time output: {result.stdout}", file=sys.stderr)
    if not times:
        raise RuntimeError("No successful runs")
    stats = {
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "runs": len(times),
        "all_times": times
    }
    return stats


def measure_kernel_performance(kernel_ll_path: str, input_size: int, repeat: int = 10) -> Dict[str, Any]:
    """Full pipeline: infer type, generate host, compile, run, return stats."""
    kernel_type = infer_kernel_type(kernel_ll_path)
    host_code = generate_host_code(kernel_type, input_size)
    exe_path = os.path.abspath("perf_test_exe")
    if not compile_and_link(kernel_ll_path, host_code, exe_path):
        raise RuntimeError("Compilation or linking failed")
    stats = run_executable(exe_path, repeat)
    os.remove(exe_path)
    return stats

# ===============================
# CLI
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Measure runtime of a tuned kernel (LLVM IR)")
    parser.add_argument('--kernel-ll', required=True, help='Path to tuned kernel LLVM IR file')
    parser.add_argument('--input-size', type=int, required=True, help='Input size for buffer allocation')
    parser.add_argument('--repeat', type=int, default=10, help='Number of runs for timing statistics')
    args = parser.parse_args()
    try:
        stats = measure_kernel_performance(args.kernel_ll, args.input_size, args.repeat)
        print("Performance statistics:")
        print(f"  Runs: {stats['runs']}")
        print(f"  Min: {stats['min']:.3f} ms")
        print(f"  Max: {stats['max']:.3f} ms")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  Std: {stats['stdev']:.3f} ms")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
