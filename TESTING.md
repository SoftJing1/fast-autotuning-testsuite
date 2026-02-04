# Comprehensive Testing Plan for 1000 Kernel Variants

## Overview

This document outlines a production-grade testing strategy for validating 1000 different tuning parameter variants across gaussian and gemm kernels.

## Architecture

### Current Implementation: Hybrid CMake + Python

```
┌─────────────────────────────────────────┐
│   CMake Build System (Single Pass)      │
│  - Builds gaussian & gemm executables   │
│  - Links against OpenCL                 │
│  - Compiles with default tuning params  │
└──────────────────┬──────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│   Python Test Framework (pytest)        │
│  - Discovers 1000+ tuning variants      │
│  - Runs each kernel with variant set    │
│  - Collects performance metrics         │
│  - Generates reports & analysis         │
└─────────────────────────────────────────┘
```

## Testing Strategies

### **Strategy 1: Python-Based Variant Testing (RECOMMENDED)**

**Best for:** Flexible testing, quick iteration, CI/CD integration

**Files:**
- `tests/test_kernels.py` - Main testing framework
- `tests/conftest.py` - Pytest configuration

**How to Run:**

```bash
# Quick test (first 10 variants)
cd build
python ../tests/test_kernels.py

# Full test with pytest (all discovered variants)
pytest ../tests/test_kernels.py -v

# Test specific kernel type
pytest ../tests/test_kernels.py::test_gaussian_kernel -v

# Parallel execution (4 workers)
pytest ../tests/test_kernels.py -n 4

# Generate HTML report
pytest ../tests/test_kernels.py --html=report.html
```

**Advantages:**
- ✅ Single build (1 minute vs 1000+ minutes for separate builds)
- ✅ Automatic variant discovery (finds all .cl files)
- ✅ Flexible test organization
- ✅ Easy performance profiling
- ✅ CI/CD friendly
- ✅ JSON result export for analysis

**Output:**
```
test_results.json
├── kernel_type: "gaussian"
├── variant_id: 42
├── success: true
├── runtime_ms: 1234.5
└── error: ""
```

---

### **Strategy 2: CMake-Based Testing**

**Best for:** Integration with CMake build pipeline

**How to Run:**

```bash
# Build and run Python tests
cmake --build . && make test_python

# Or use ctest
ctest --verbose
```

**Features:**
- Integrates with standard CMake workflow
- Custom target `test_python` in CMakeLists.txt
- Easy inclusion in CI/CD pipelines

---

### **Strategy 3: Per-Variant Build Testing (Advanced)**

**Best for:** Validating compile-time tuning parameters

This approach would build separate executables for each variant:

```cmake
# For each tuning parameter file
foreach(VARIANT IN LISTS GAUSSIAN_VARIANTS)
    add_executable(gaussian_${VARIANT} kernels/kernel-host/gaussian_host.cc)
    target_compile_definitions(gaussian_${VARIANT} PRIVATE 
        TUNING_HEADER=<${VARIANT}>)
    add_test(NAME gaussian_${VARIANT} COMMAND gaussian_${VARIANT})
endforeach()
```

**Pros:** Validates compile-time behavior, full optimization
**Cons:** 1000+ builds (~16+ hours), 1000+ executables, high disk usage

**Recommendation:** Use only for critical variants (e.g., top 50)

---

### **Strategy 4: Runtime Parameter Loading (Future Enhancement)**

Modify C++ code to accept tuning header paths as runtime arguments:

```cpp
./gaussian --tuning-header /path/to/gaussian_1024x1024_000042.cl
```

**Benefits:**
- True runtime flexibility
- Single executable for all variants
- Minimal build time

**Implementation:**
1. Use environment variables or JSON config
2. Dynamically include tuning headers
3. Validate parameters at runtime

---

## Testing Metrics & Analysis

### Metrics Collected

```python
{
    "total_variants": 1000,
    "gaussian_variants": 500,
    "gemm_variants": 500,
    "passed": 998,
    "failed": 2,
    "skipped": 0,
    "pass_rate": 99.8,
    "avg_runtime_ms": 125.4,
    "min_runtime_ms": 89.2,
    "max_runtime_ms": 456.7,
    "median_runtime_ms": 110.5,
    "total_time_minutes": 45.3
}
```

### Performance Analysis

```bash
# Generate performance report
python -c "
import json
with open('test_results.json') as f:
    results = json.load(f)
    
gaussian = [r for r in results if r['kernel_type'] == 'gaussian']
print(f'Gaussian avg: {sum(r[\"runtime_ms\"] for r in gaussian) / len(gaussian):.1f}ms')
"
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Kernel Testing

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: |
          mkdir build && cd build
          cmake .. && cmake --build .
      - name: Run Tests (Quick)
        run: |
          cd build
          python ../tests/test_kernels.py
      - name: Run Tests (Full with pytest)
        run: |
          pip install pytest pytest-html
          pytest ../tests/test_kernels.py --html=report.html
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_results.json
```

---

## Recommended Testing Workflow

### Phase 1: Quick Validation (5 minutes)
```bash
cd build
python ../tests/test_kernels.py  # Tests first 10 variants
```

### Phase 2: Subset Testing (15 minutes)
```bash
pytest ../tests/test_kernels.py -k "variant_00" -v  # Test variants 0-9
```

### Phase 3: Full Suite (1-2 hours)
```bash
pytest ../tests/test_kernels.py -v --tb=short
```

### Phase 4: Analysis
```bash
python analyze_results.py test_results.json
```

---

## Implementation Checklist

- [x] Create `tests/test_kernels.py` with KernelTestRunner class
- [x] Implement automatic variant discovery
- [x] Add pytest parametrization
- [x] Create JSON result export
- [x] Update CMakeLists.txt with testing configuration
- [ ] Create `tests/conftest.py` for shared fixtures
- [ ] Create `analyze_results.py` for post-test analysis
- [ ] Add performance comparison script
- [ ] Create HTML report generation
- [ ] Setup CI/CD pipeline

---

## Future Enhancements

### 1. Performance Benchmarking
```python
# Identify best/worst performing variants
best_variant = max(results, key=lambda r: 1/r['runtime_ms'])
worst_variant = max(results, key=lambda r: r['runtime_ms'])
```

### 2. Regression Testing
```python
# Compare against baseline
baseline = load_baseline()
regressions = [r for r in results if r['runtime_ms'] > baseline[r['variant_id']] * 1.1]
```

### 3. Parallel Execution
```bash
# Run up to 4 tests in parallel
pytest ../tests/test_kernels.py -n 4
```

### 4. Timeout Management
```python
# Adaptive timeout based on variant size
timeout = base_timeout * (size_factor ** 0.5)
```

### 5. Resource Profiling
```python
# Measure GPU memory, compute utilization
profile = profile_kernel(variant)
metrics = {
    'memory_mb': profile.peak_memory,
    'compute_utilization': profile.avg_compute,
    'memory_bandwidth': profile.bandwidth
}
```

---

## Troubleshooting

### Issue: Tests timeout
**Solution:** Increase timeout in KernelTestRunner.run_test()
```python
result = runner.run_test(kernel_type, variant_id, timeout_sec=120)
```

### Issue: Some variants not discovered
**Solution:** Check tuning parameter directory structure:
```bash
ls kernels/tuning_params/gaussian/
ls kernels/tuning_params/gemm/
```

### Issue: GPU out of memory
**Solution:** Run tests sequentially or on smaller datasets:
```python
# Modify gaussian_host.cc to accept size parameter
const size_t H = 512, W = 512;  // Reduce from 1024
```

---

## File Reference

| File | Purpose |
|------|---------|
| `tests/test_kernels.py` | Main testing framework, variant discovery |
| `CMakeLists.txt` | Build configuration with testing targets |
| `kernels/kernel-host/gaussian_host.cc` | Gaussian kernel with arg support |
| `kernels/kernel-host/gemm_host.cc` | GEMM kernel implementation |
| `test_results.json` | Generated test results (JSON format) |

---

## Questions & Support

For issues or questions:
1. Check test_results.json for detailed error messages
2. Review CMake build output for compilation errors
3. Verify GPU availability: `clinfo`
4. Check OpenCL installation
