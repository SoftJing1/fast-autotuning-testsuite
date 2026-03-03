#!/usr/bin/env python3
"""
Comprehensive testing framework for 1000 tuning parameter variants.
Tests both gaussian and gemm kernels with all available tuning parameters.
"""

import os
import sys
import subprocess
import glob
import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import time


@dataclass
class TestResult:
    """Store results of a single kernel test"""
    kernel_type: str  # 'gaussian' or 'gemm'
    param_file: str
    variant_id: int
    success: bool
    runtime_ms: float
    error_msg: str = ""
    stdout: str = ""


class KernelTestRunner:
    """Manages building and testing kernel variants"""
    
    def __init__(self, build_dir: Optional[str] = None, source_dir: Optional[str] = None):
        # Default paths
        if source_dir is None:
            source_dir = str(Path(__file__).parent.parent)
        if build_dir is None:
            build_dir = str(Path(source_dir) / "build")
            
        self.source_dir = Path(source_dir)
        self.build_dir = Path(build_dir)
        self.gaussian_tuning_dir = self.source_dir / "kernels" / "tuning_params" / "gaussian"
        self.gemm_tuning_dir = self.source_dir / "kernels" / "tuning_params" / "gemm"
        self.gaussian_exe = self.build_dir / "gaussian"
        self.gemm_exe = self.build_dir / "gemm"
        
    def discover_tuning_parameters(self, kernel_type: str) -> Dict[int, Path]:
        """
        Discover all tuning parameter files for a given kernel type.
        Returns dict mapping variant_id -> file_path
        """
        if kernel_type == "gaussian":
            param_dir = self.gaussian_tuning_dir
            pattern = "gaussian_*.json"
        elif kernel_type == "gemm":
            param_dir = self.gemm_tuning_dir
            pattern = "gemm_*.json"
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        files = sorted(glob.glob(str(param_dir / pattern)))
        
        # Extract variant ID from filename (e.g., gaussian_1024x1024_000042.json -> 42)
        variants = {}
        for filepath in files:
            filename = Path(filepath).name
            # Extract last number group (variant ID)
            parts = filename.replace('.json', '').split('_')
            try:
                variant_id = int(parts[-1])
                variants[variant_id] = Path(filepath)
            except (ValueError, IndexError):
                print(f"Warning: Could not extract variant ID from {filename}")
        
        return variants
    
    def build_kernel_variant(self, kernel_type: str, tuning_param_file: Path) -> bool:
        """
        Rebuild kernel with specific tuning parameter file.
        This requires modifying CMakeLists.txt and rebuilding.
        For efficiency, we recommend using a single executable approach instead.
        """
        # For now, we assume the prebuilt gaussian and gemm executables exist
        # A future optimization could involve refactoring to enable per-test compilation
        return True
    
    def run_test(self, kernel_type: str, variant_id: int, 
                 param_file: Path, timeout_sec: int = 60) -> TestResult:
        """
        Run a single kernel test with given JSON configuration file.
        The executable now accepts the JSON config path as a command-line argument.
        """
        exe_path = self.gaussian_exe if kernel_type == "gaussian" else self.gemm_exe
        
        if not exe_path.exists():
            return TestResult(
                kernel_type=kernel_type,
                param_file=str(param_file),
                variant_id=variant_id,
                success=False,
                runtime_ms=0.0,
                error_msg=f"Executable not found: {exe_path}"
            )
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [str(exe_path), str(param_file)],  # Pass JSON config file as argument
                cwd=str(self.build_dir),
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            success = result.returncode == 0
            error_msg = result.stderr if result.returncode != 0 else ""
            
            return TestResult(
                kernel_type=kernel_type,
                param_file=str(param_file),
                variant_id=variant_id,
                success=success,
                runtime_ms=elapsed_ms,
                error_msg=error_msg,
                stdout=result.stdout[:500] if result.stdout else ""
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                kernel_type=kernel_type,
                param_file=str(param_file),
                variant_id=variant_id,
                success=False,
                runtime_ms=timeout_sec * 1000,
                error_msg=f"Timeout after {timeout_sec}s"
            )
        except Exception as e:
            return TestResult(
                kernel_type=kernel_type,
                param_file=str(param_file),
                variant_id=variant_id,
                success=False,
                runtime_ms=0.0,
                error_msg=str(e)
            )
    
    def run_all_tests(self, kernel_type: Optional[str] = None, 
                     max_variants: Optional[int] = None) -> List[TestResult]:
        """
        Run all discovered tuning parameter variants.
        
        Args:
            kernel_type: 'gaussian', 'gemm', or None for both
            max_variants: Limit testing to first N variants (for quick testing)
        
        Returns:
            List of TestResult objects
        """
        results = []
        kernel_types = [kernel_type] if kernel_type else ["gaussian", "gemm"]
        
        for ktype in kernel_types:
            variants = self.discover_tuning_parameters(ktype)
            variant_ids = sorted(variants.keys())
            
            if max_variants:
                variant_ids = variant_ids[:max_variants]
            
            print(f"Testing {ktype} with {len(variant_ids)} variants...")
            
            for vid in variant_ids:
                param_file = variants[vid]
                result = self.run_test(ktype, vid, param_file)
                results.append(result)
                
                status = "✓" if result.success else "✗"
                print(f"  {status} {ktype}_{vid:06d}: {result.runtime_ms:.1f}ms")
        
        return results


# ============================================================================
# Pytest Integration
# ============================================================================

# Global test runner instance
_runner = None

def get_runner():
    """Get or create the global test runner"""
    global _runner
    if _runner is None:
        _runner = KernelTestRunner()
    return _runner


@pytest.fixture(scope="session")
def runner():
    """Pytest fixture for kernel test runner"""
    return get_runner()


@pytest.fixture(params=[
    (0, "gaussian"),
    (1, "gaussian"),
    (2, "gaussian"),
    (3, "gaussian"),
    (4, "gaussian"),
    (5, "gaussian"),
    # Add more for comprehensive testing, or use dynamic discovery
], ids=lambda x: f"{x[1]}_variant_{x[0]:06d}")
def gaussian_variants(request):
    """Parametrized test for gaussian kernel variants"""
    variant_id, kernel_type = request.param
    runner = get_runner()
    variants = runner.discover_tuning_parameters(kernel_type)
    
    if variant_id not in variants:
        pytest.skip(f"Variant {variant_id} not found")
    
    return kernel_type, variant_id, variants[variant_id]


@pytest.fixture(params=[
    (0, "gemm"),
    (1, "gemm"),
], ids=lambda x: f"{x[1]}_variant_{x[0]:06d}")
def gemm_variants(request):
    """Parametrized test for gemm kernel variants"""
    variant_id, kernel_type = request.param
    runner = get_runner()
    variants = runner.discover_tuning_parameters(kernel_type)
    
    if variant_id not in variants:
        pytest.skip(f"Variant {variant_id} not found")
    
    return kernel_type, variant_id, variants[variant_id]


def test_gaussian_kernel(gaussian_variants):
    """Test gaussian kernel with a specific variant"""
    runner = get_runner()
    kernel_type, variant_id, param_file = gaussian_variants
    
    result = runner.run_test(kernel_type, variant_id, param_file)
    
    assert result.success, f"Kernel failed: {result.error_msg}"
    assert result.runtime_ms > 0, "Runtime not measured"


def test_gemm_kernel(gemm_variants):
    """Test gemm kernel with a specific variant"""
    runner = get_runner()
    kernel_type, variant_id, param_file = gemm_variants
    
    result = runner.run_test(kernel_type, variant_id, param_file)
    
    assert result.success, f"Kernel failed: {result.error_msg}"
    assert result.runtime_ms > 0, "Runtime not measured"


# ============================================================================
# Command-line Testing
# ============================================================================

def main():
    """Standalone testing mode (can be run without pytest)"""
    runner = KernelTestRunner()
    
    # For quick testing, limit to first 10 variants
    print("=== Quick Test (10 variants per kernel) ===\n")
    results = runner.run_all_tests(max_variants=100)
    
    # Summary
    gaussian_results = [r for r in results if r.kernel_type == "gaussian"]
    gemm_results = [r for r in results if r.kernel_type == "gemm"]
    
    print("\n=== Test Summary ===")
    print(f"Gaussian: {sum(1 for r in gaussian_results if r.success)}/{len(gaussian_results)} passed")
    print(f"GEMM:     {sum(1 for r in gemm_results if r.success)}/{len(gemm_results)} passed")
    
    # Save results to JSON
    results_file = Path(__file__).parent / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump([{
            'kernel_type': r.kernel_type,
            'variant_id': r.variant_id,
            'success': r.success,
            'runtime_ms': r.runtime_ms,
            'error': r.error_msg
        } for r in results], f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return exit code based on success
    failed = sum(1 for r in results if not r.success)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
