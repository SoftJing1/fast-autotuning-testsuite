================================================================================
GAUSSIAN KERNEL HOST CODE - Minimal Execution Framework
================================================================================

DESCRIPTION
-----------
This directory contains a minimal OpenCL host program to execute the Gaussian
blur kernel (gaussian_static_1.cl) with pre-tuned parameters and result
validation.

The host code:
- Loads and compiles the Gaussian blur kernel template
- Applies pre-computed tuning parameters via compile-time defines
- Executes the kernel on GPU with random input data
- Validates GPU results against CPU reference implementation
- Measures kernel execution time

COMPONENTS
----------
gaussian_host.cc    - Main host program (standalone, no ATF dependency)
Makefile            - Build configuration using clang++
generate_golden.py  - Python script to generate golden reference results

INPUT/OUTPUT
------------
Input:   Padded image of size (H+4) × (W+4) with random values [10.0, 100.0]
Output:  Filtered image of size H × W
Kernel:  5×5 convolution filter with fixed coefficients:
         [2  4  5  4  2]
         [4  9 12  9  4]
         [5 12 15 12  5]
         [4  9 12  9  4]
         [2  4  5  4  2]

COMPILATION
-----------
$ make              # Compile using clang++
$ make run          # Compile and run
$ make clean        # Remove build artifacts

DEPENDENCIES
------------
- OpenCL 1.2+ with GPU support
- clang++ compiler (or modify Makefile for g++)
- libOpenCL

TUNING PARAMETERS
-----------------
Default parameters from: ../tuning_params/gaussian/gaussian_1024x1024_000000.cl

Key parameters:
- INPUT_SIZE_1=1024, INPUT_SIZE_2=1024  (Image dimensions)
- GLB_1, WG_1, LCL_1, WI_1, PRV_1       (Dimension 1 workload distribution)
- GLB_2, WG_2, LCL_2, WI_2, PRV_2       (Dimension 2 workload distribution)
- Cache flags (IMAGES_CACHE_*, FILTER_CACHE_*, OUT_CACHE_*)

Modify these #defines in gaussian_host.cc to test different configurations.

VALIDATION
----------
The program automatically validates GPU results by:
1. Computing reference result on CPU using the same 5×5 filter
2. Comparing GPU output with CPU reference (tolerance: 1e-4f)
3. Reporting any mismatches with detailed error information

Expected output on success:
  ✓ Result is CORRECT (max diff: X.XXXXX)

MODIFICATIONS FOR DIFFERENT SIZES
----------------------------------
To test different image sizes, modify in gaussian_host.cc:
  const size_t H = 1024, W = 1024;  // Change to desired size
  
Note: Must ensure matching tuning parameters exist or update the #defines.

PERFORMANCE MEASUREMENT
-----------------------
The program outputs GPU kernel execution time in milliseconds.
This is the actual kernel runtime only (excludes memory transfers and CPU overhead).

NOTES
-----
- This is a minimal implementation without tuning/search overhead
- GPU device selection is hardcoded to first GPU (platform 0, device 0)
- Modify clGetDeviceIDs() call for device selection on multi-GPU systems
- Input is generated with random seed for reproducibility use fixed_seed
================================================================================
