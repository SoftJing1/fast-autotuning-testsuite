#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "tc_8x8x8x8x8x8x8_000002.cl"  // Tuning parameters

std::string read_kernel_source(const char *filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// CPU reference: Simplified tensor contraction
// Contracts tensor A[M1xM2xM3xM4xM5xM6] with tensor B[N1] to produce output C
void compute_reference(const std::vector<float> &a, const std::vector<float> &b,
                       std::vector<float> &c,
                       size_t M1, size_t M2, size_t M3, size_t M4, size_t M5, size_t M6, size_t N1) {
    size_t output_size = M1 * M2 * M3 * M4 * M5 * M6;
    
    std::fill(c.begin(), c.end(), 0.0f);
    
    // Tensor contraction: sum over reduction dimension N1
    for (size_t idx = 0; idx < output_size; ++idx) {
        for (size_t n = 0; n < N1; ++n) {
            // Simplified contraction: element-wise multiplication and accumulation
            c[idx] += a[idx * N1 + n] * b[n];
        }
    }
}

// Reduce intermediate results when multiple workgroups were used
void reduce_intermediate_results(const std::vector<float> &int_res,
                                   std::vector<float> &res_g,
                                   size_t output_elements,
                                   size_t num_wg_r_1) {
    std::fill(res_g.begin(), res_g.end(), 0.0f);
    
    for (size_t wg = 0; wg < num_wg_r_1; ++wg) {
        for (size_t i = 0; i < output_elements; ++i) {
            res_g[i] += int_res[wg * output_elements + i];
        }
    }
}

// Compare GPU result with reference (with floating-point tolerance)
bool validate_result(const std::vector<float> &gpu_out, const std::vector<float> &ref_out,
                     size_t size) {
    int mismatch_count = 0;
    float max_diff = 0.0f;
    const float tolerance = 0.1f;  // Higher tolerance for complex tensor operations
    
    std::cout << "Validating computation..." << std::endl;
    
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(gpu_out[i] - ref_out[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > tolerance) {
            mismatch_count++;
            if (mismatch_count <= 5) {  // Print first 5 mismatches
                std::cout << "  Mismatch at index " << i << std::endl;
                std::cout << "    Reference: " << ref_out[i] << std::endl;
                std::cout << "    GPU:       " << gpu_out[i] << std::endl;
                std::cout << "    Diff:      " << diff << std::endl;
            }
        }
    }
    
    if (mismatch_count == 0) {
        std::cout << "✓ Result is CORRECT (max diff: " << max_diff << ")" << std::endl;
        return true;
    } else {
        std::cout << "✗ Result is INCORRECT" << std::endl;
        std::cout << "  Mismatches > " << tolerance << ": " << mismatch_count << "/" << size << std::endl;
        std::cout << "  Max difference: " << max_diff << std::endl;
        return false;
    }
}

int main() {
    const size_t M1 = 8, M2 = 8, M3 = 8, M4 = 8, M5 = 8, M6 = 8, N1 = 8;
    const size_t output_elements = M1 * M2 * M3 * M4 * M5 * M6;
    const size_t a_total = output_elements * N1;  // A is input tensor that gets contracted
    const size_t b_total = N1;  // B is the contraction vector
    const size_t c_total = output_elements;  // C is the output
    
    cl_int err;

    // Get platform and device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Create context and command queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context: " << err << std::endl;
        return EXIT_FAILURE;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating command queue: " << err << std::endl;
        return EXIT_FAILURE;
    }

    // Create buffers with random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.0);  // Smaller values for stability
    
    std::vector<float> a(a_total);
    std::vector<float> b(b_total);
    std::vector<float> c(c_total, 0.0f);
    
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(dis(gen));
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(dis(gen));

    // Calculate res_g size based on destination levels (follows ATF pattern)
    size_t res_g_size = output_elements;
    if (G_CB_RES_DEST_LEVEL == 2) {
        res_g_size *= NUM_WG_R_1;
    }
    if (L_CB_RES_DEST_LEVEL == 2) {
        res_g_size *= NUM_WI_R_1;
    }
    
    // Calculate int_res size for multi-workgroup reduction
    size_t int_res_size = output_elements * NUM_WG_R_1;
    
    std::vector<float> res_g(res_g_size, 0.0f);
    std::vector<float> int_res(int_res_size, 0.0f);
    
    std::cout << "Problem size: " << M1 << "x" << M2 << "x" << M3 << "x" << M4 
              << "x" << M5 << "x" << M6 << "x" << N1 << std::endl;
    std::cout << "Input A size: " << a_total << " floats (" << (a_total * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Input B size: " << b_total << " floats" << std::endl;
    std::cout << "Output C size: " << c_total << " floats" << std::endl;
    std::cout << "res_g size: " << res_g_size << " floats" << std::endl;
    std::cout << "int_res size: " << int_res_size << " floats" << std::endl;
    std::cout << "NUM_WG_R_1: " << NUM_WG_R_1 << std::endl;
    
    cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   a.size() * sizeof(float), a.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating buffer A: " << err << std::endl;
        return EXIT_FAILURE;
    }
    
    cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   b.size() * sizeof(float), b.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating buffer B: " << err << std::endl;
        return EXIT_FAILURE;
    }
    
    cl_mem buf_res_g = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      res_g_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating buffer res_g: " << err << std::endl;
        return EXIT_FAILURE;
    }
    
    cl_mem buf_int_res = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        int_res_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating buffer int_res: " << err << std::endl;
        return EXIT_FAILURE;
    }
    
    cl_mem buf_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   c.size() * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating buffer C: " << err << std::endl;
        return EXIT_FAILURE;
    }


    // ===== KERNEL 1: Main computation =====
    std::cout << "\n=== Compiling and running tc_1 kernel ===" << std::endl;
    
    // Load and compile kernel_1
    std::string kernel1_source = read_kernel_source("./tc_abcdef_gebc_dfga_1.cl");
    const char *src1 = kernel1_source.c_str();
    cl_program program1 = clCreateProgramWithSource(context, 1, &src1, nullptr, &err);

    std::string compile_flags = "-DTYPE_T=float -DTYPE_TS=float "
        "-DCACHE_L_CB=" + std::to_string(CACHE_L_CB) +
        " -DCACHE_P_CB=" + std::to_string(CACHE_P_CB) +
        " -DG_CB_RES_DEST_LEVEL=" + std::to_string(G_CB_RES_DEST_LEVEL) +
        " -DL_CB_RES_DEST_LEVEL=" + std::to_string(L_CB_RES_DEST_LEVEL) +
        " -DP_CB_RES_DEST_LEVEL=" + std::to_string(P_CB_RES_DEST_LEVEL) +
        " -DINPUT_SIZE_L_1=" + std::to_string(INPUT_SIZE_L_1) +
        " -DL_CB_SIZE_L_1=" + std::to_string(L_CB_SIZE_L_1) +
        " -DP_CB_SIZE_L_1=" + std::to_string(P_CB_SIZE_L_1) +
        " -DOCL_DIM_L_1=" + std::to_string(OCL_DIM_L_1) +
        " -DNUM_WG_L_1=" + std::to_string(NUM_WG_L_1) +
        " -DNUM_WI_L_1=" + std::to_string(NUM_WI_L_1) +
        " -DINPUT_SIZE_L_2=" + std::to_string(INPUT_SIZE_L_2) +
        " -DL_CB_SIZE_L_2=" + std::to_string(L_CB_SIZE_L_2) +
        " -DP_CB_SIZE_L_2=" + std::to_string(P_CB_SIZE_L_2) +
        " -DOCL_DIM_L_2=" + std::to_string(OCL_DIM_L_2) +
        " -DNUM_WG_L_2=" + std::to_string(NUM_WG_L_2) +
        " -DNUM_WI_L_2=" + std::to_string(NUM_WI_L_2) +
        " -DINPUT_SIZE_L_3=" + std::to_string(INPUT_SIZE_L_3) +
        " -DL_CB_SIZE_L_3=" + std::to_string(L_CB_SIZE_L_3) +
        " -DP_CB_SIZE_L_3=" + std::to_string(P_CB_SIZE_L_3) +
        " -DOCL_DIM_L_3=" + std::to_string(OCL_DIM_L_3) +
        " -DNUM_WG_L_3=" + std::to_string(NUM_WG_L_3) +
        " -DNUM_WI_L_3=" + std::to_string(NUM_WI_L_3) +
        " -DINPUT_SIZE_L_4=" + std::to_string(INPUT_SIZE_L_4) +
        " -DL_CB_SIZE_L_4=" + std::to_string(L_CB_SIZE_L_4) +
        " -DP_CB_SIZE_L_4=" + std::to_string(P_CB_SIZE_L_4) +
        " -DOCL_DIM_L_4=" + std::to_string(OCL_DIM_L_4) +
        " -DNUM_WG_L_4=" + std::to_string(NUM_WG_L_4) +
        " -DNUM_WI_L_4=" + std::to_string(NUM_WI_L_4) +
        " -DINPUT_SIZE_L_5=" + std::to_string(INPUT_SIZE_L_5) +
        " -DL_CB_SIZE_L_5=" + std::to_string(L_CB_SIZE_L_5) +
        " -DP_CB_SIZE_L_5=" + std::to_string(P_CB_SIZE_L_5) +
        " -DOCL_DIM_L_5=" + std::to_string(OCL_DIM_L_5) +
        " -DNUM_WG_L_5=" + std::to_string(NUM_WG_L_5) +
        " -DNUM_WI_L_5=" + std::to_string(NUM_WI_L_5) +
        " -DINPUT_SIZE_L_6=" + std::to_string(INPUT_SIZE_L_6) +
        " -DL_CB_SIZE_L_6=" + std::to_string(L_CB_SIZE_L_6) +
        " -DP_CB_SIZE_L_6=" + std::to_string(P_CB_SIZE_L_6) +
        " -DOCL_DIM_L_6=" + std::to_string(OCL_DIM_L_6) +
        " -DNUM_WG_L_6=" + std::to_string(NUM_WG_L_6) +
        " -DNUM_WI_L_6=" + std::to_string(NUM_WI_L_6) +
        " -DINPUT_SIZE_R_1=" + std::to_string(INPUT_SIZE_R_1) +
        " -DL_CB_SIZE_R_1=" + std::to_string(L_CB_SIZE_R_1) +
        " -DP_CB_SIZE_R_1=" + std::to_string(P_CB_SIZE_R_1) +
        " -DOCL_DIM_R_1=" + std::to_string(OCL_DIM_R_1) +
        " -DNUM_WG_R_1=" + std::to_string(NUM_WG_R_1) +
        " -DNUM_WI_R_1=" + std::to_string(NUM_WI_R_1) +
        " -DL_REDUCTION=" + std::to_string(L_REDUCTION) +
        " -DP_WRITE_BACK=" + std::to_string(P_WRITE_BACK) +
        " -DL_WRITE_BACK=" + std::to_string(L_WRITE_BACK);

    err = clBuildProgram(program1, 1, &device, compile_flags.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program1, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program1, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Compilation error for tc_1:\n" << log.data() << std::endl;
        return EXIT_FAILURE;
    }

    // Create kernel_1
    cl_kernel kernel1 = clCreateKernel(program1, "tc_1", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel tc_1: " << err << std::endl;
        return EXIT_FAILURE;
    }

    // Set kernel_1 arguments: (a, b, res_g, int_res)
    clSetKernelArg(kernel1, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel1, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel1, 2, sizeof(cl_mem), &buf_res_g);
    clSetKernelArg(kernel1, 3, sizeof(cl_mem), &buf_int_res);

    // Calculate global and local sizes for kernel_1 (7D work space)
    size_t global_size_1[3], local_size_1[3];
    
    // Map 7 logical dimensions to 3 OpenCL dimensions
    // Dimensions 0-1 map to x,y; dimensions 2-6 multiply together for z
    global_size_1[0] = 1;
    global_size_1[1] = 1;
    global_size_1[2] = 1;
    local_size_1[0] = 1;
    local_size_1[1] = 1;
    local_size_1[2] = 1;
    
    // Map each logical dimension to OCL dimension based on OCL_DIM_* parameters
    if (OCL_DIM_L_1 == 0) {
        global_size_1[0] *= NUM_WG_L_1 * NUM_WI_L_1;
        local_size_1[0] *= NUM_WI_L_1;
    } else if (OCL_DIM_L_1 == 1) {
        global_size_1[1] *= NUM_WG_L_1 * NUM_WI_L_1;
        local_size_1[1] *= NUM_WI_L_1;
    } else {
        global_size_1[2] *= NUM_WG_L_1 * NUM_WI_L_1;
        local_size_1[2] *= NUM_WI_L_1;
    }
    
    if (OCL_DIM_L_2 == 0) {
        global_size_1[0] *= NUM_WG_L_2 * NUM_WI_L_2;
        local_size_1[0] *= NUM_WI_L_2;
    } else if (OCL_DIM_L_2 == 1) {
        global_size_1[1] *= NUM_WG_L_2 * NUM_WI_L_2;
        local_size_1[1] *= NUM_WI_L_2;
    } else {
        global_size_1[2] *= NUM_WG_L_2 * NUM_WI_L_2;
        local_size_1[2] *= NUM_WI_L_2;
    }
    
    if (OCL_DIM_L_3 == 0) {
        global_size_1[0] *= NUM_WG_L_3 * NUM_WI_L_3;
        local_size_1[0] *= NUM_WI_L_3;
    } else if (OCL_DIM_L_3 == 1) {
        global_size_1[1] *= NUM_WG_L_3 * NUM_WI_L_3;
        local_size_1[1] *= NUM_WI_L_3;
    } else {
        global_size_1[2] *= NUM_WG_L_3 * NUM_WI_L_3;
        local_size_1[2] *= NUM_WI_L_3;
    }
    
    if (OCL_DIM_L_4 == 0) {
        global_size_1[0] *= NUM_WG_L_4 * NUM_WI_L_4;
        local_size_1[0] *= NUM_WI_L_4;
    } else if (OCL_DIM_L_4 == 1) {
        global_size_1[1] *= NUM_WG_L_4 * NUM_WI_L_4;
        local_size_1[1] *= NUM_WI_L_4;
    } else {
        global_size_1[2] *= NUM_WG_L_4 * NUM_WI_L_4;
        local_size_1[2] *= NUM_WI_L_4;
    }
    
    if (OCL_DIM_L_5 == 0) {
        global_size_1[0] *= NUM_WG_L_5 * NUM_WI_L_5;
        local_size_1[0] *= NUM_WI_L_5;
    } else if (OCL_DIM_L_5 == 1) {
        global_size_1[1] *= NUM_WG_L_5 * NUM_WI_L_5;
        local_size_1[1] *= NUM_WI_L_5;
    } else {
        global_size_1[2] *= NUM_WG_L_5 * NUM_WI_L_5;
        local_size_1[2] *= NUM_WI_L_5;
    }
    
    if (OCL_DIM_L_6 == 0) {
        global_size_1[0] *= NUM_WG_L_6 * NUM_WI_L_6;
        local_size_1[0] *= NUM_WI_L_6;
    } else if (OCL_DIM_L_6 == 1) {
        global_size_1[1] *= NUM_WG_L_6 * NUM_WI_L_6;
        local_size_1[1] *= NUM_WI_L_6;
    } else {
        global_size_1[2] *= NUM_WG_L_6 * NUM_WI_L_6;
        local_size_1[2] *= NUM_WI_L_6;
    }
    
    if (OCL_DIM_R_1 == 0) {
        global_size_1[0] *= NUM_WG_R_1 * NUM_WI_R_1;
        local_size_1[0] *= NUM_WI_R_1;
    } else if (OCL_DIM_R_1 == 1) {
        global_size_1[1] *= NUM_WG_R_1 * NUM_WI_R_1;
        local_size_1[1] *= NUM_WI_R_1;
    } else {
        global_size_1[2] *= NUM_WG_R_1 * NUM_WI_R_1;
        local_size_1[2] *= NUM_WI_R_1;
    }

    std::cout << "Kernel 1 NDRange:" << std::endl;
    std::cout << "  Global: " << global_size_1[0] << " x " << global_size_1[1] << " x " << global_size_1[2] << std::endl;
    std::cout << "  Local:  " << local_size_1[0] << " x " << local_size_1[1] << " x " << local_size_1[2] << std::endl;

    // Execute kernel_1
    cl_event event1;
    err = clEnqueueNDRangeKernel(queue, kernel1, 3, nullptr, global_size_1, local_size_1, 0, nullptr, &event1);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel tc_1: " << err << std::endl;
        return EXIT_FAILURE;
    }
    clWaitForEvents(1, &event1);
    
    cl_ulong start_time_1, end_time_1;
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time_1, nullptr);
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time_1, nullptr);
    std::cout << "✓ tc_1 executed in " << ((end_time_1 - start_time_1) / 1000000.0) << " ms" << std::endl;

    // ===== KERNEL 2: Reduction (if needed) =====
    bool needs_reduction = (NUM_WG_R_1 > 1);
    
    if (needs_reduction) {
        std::cout << "\n=== Compiling and running tc_2 kernel ===" << std::endl;
        
        // Load and compile kernel_2
        std::string kernel2_source = read_kernel_source("./tc_abcdef_gebc_dfga_2.cl");
        const char *src2 = kernel2_source.c_str();
        cl_program program2 = clCreateProgramWithSource(context, 1, &src2, nullptr, &err);

        err = clBuildProgram(program2, 1, &device, compile_flags.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program2, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program2, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "Compilation error for tc_2:\n" << log.data() << std::endl;
            return EXIT_FAILURE;
        }

        // Create kernel_2
        cl_kernel kernel2 = clCreateKernel(program2, "tc_2", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Error creating kernel tc_2: " << err << std::endl;
            return EXIT_FAILURE;
        }

        // Set kernel_2 arguments: (int_res, res_g, c)
        clSetKernelArg(kernel2, 0, sizeof(cl_mem), &buf_int_res);
        clSetKernelArg(kernel2, 1, sizeof(cl_mem), &buf_res_g);
        clSetKernelArg(kernel2, 2, sizeof(cl_mem), &buf_c);

        // Calculate global and local sizes for kernel_2
        // kernel_2 has no NUM_WG_R_1 in the R_1 dimension (only NUM_WI_R_1)
        size_t global_size_2[3], local_size_2[3];
        global_size_2[0] = 1;
        global_size_2[1] = 1;
        global_size_2[2] = 1;
        local_size_2[0] = 1;
        local_size_2[1] = 1;
        local_size_2[2] = 1;
        
        if (OCL_DIM_L_1 == 0) {
            global_size_2[0] *= NUM_WG_L_1 * NUM_WI_L_1;
            local_size_2[0] *= NUM_WI_L_1;
        } else if (OCL_DIM_L_1 == 1) {
            global_size_2[1] *= NUM_WG_L_1 * NUM_WI_L_1;
            local_size_2[1] *= NUM_WI_L_1;
        } else {
            global_size_2[2] *= NUM_WG_L_1 * NUM_WI_L_1;
            local_size_2[2] *= NUM_WI_L_1;
        }
        
        if (OCL_DIM_L_2 == 0) {
            global_size_2[0] *= NUM_WG_L_2 * NUM_WI_L_2;
            local_size_2[0] *= NUM_WI_L_2;
        } else if (OCL_DIM_L_2 == 1) {
            global_size_2[1] *= NUM_WG_L_2 * NUM_WI_L_2;
            local_size_2[1] *= NUM_WI_L_2;
        } else {
            global_size_2[2] *= NUM_WG_L_2 * NUM_WI_L_2;
            local_size_2[2] *= NUM_WI_L_2;
        }
        
        if (OCL_DIM_L_3 == 0) {
            global_size_2[0] *= NUM_WG_L_3 * NUM_WI_L_3;
            local_size_2[0] *= NUM_WI_L_3;
        } else if (OCL_DIM_L_3 == 1) {
            global_size_2[1] *= NUM_WG_L_3 * NUM_WI_L_3;
            local_size_2[1] *= NUM_WI_L_3;
        } else {
            global_size_2[2] *= NUM_WG_L_3 * NUM_WI_L_3;
            local_size_2[2] *= NUM_WI_L_3;
        }
        
        if (OCL_DIM_L_4 == 0) {
            global_size_2[0] *= NUM_WG_L_4 * NUM_WI_L_4;
            local_size_2[0] *= NUM_WI_L_4;
        } else if (OCL_DIM_L_4 == 1) {
            global_size_2[1] *= NUM_WG_L_4 * NUM_WI_L_4;
            local_size_2[1] *= NUM_WI_L_4;
        } else {
            global_size_2[2] *= NUM_WG_L_4 * NUM_WI_L_4;
            local_size_2[2] *= NUM_WI_L_4;
        }
        
        if (OCL_DIM_L_5 == 0) {
            global_size_2[0] *= NUM_WG_L_5 * NUM_WI_L_5;
            local_size_2[0] *= NUM_WI_L_5;
        } else if (OCL_DIM_L_5 == 1) {
            global_size_2[1] *= NUM_WG_L_5 * NUM_WI_L_5;
            local_size_2[1] *= NUM_WI_L_5;
        } else {
            global_size_2[2] *= NUM_WG_L_5 * NUM_WI_L_5;
            local_size_2[2] *= NUM_WI_L_5;
        }
        
        if (OCL_DIM_L_6 == 0) {
            global_size_2[0] *= NUM_WG_L_6 * NUM_WI_L_6;
            local_size_2[0] *= NUM_WI_L_6;
        } else if (OCL_DIM_L_6 == 1) {
            global_size_2[1] *= NUM_WG_L_6 * NUM_WI_L_6;
            local_size_2[1] *= NUM_WI_L_6;
        } else {
            global_size_2[2] *= NUM_WG_L_6 * NUM_WI_L_6;
            local_size_2[2] *= NUM_WI_L_6;
        }
        
        // For R_1 dimension in kernel2, only use NUM_WI_R_1 (no NUM_WG_R_1)
        if (OCL_DIM_R_1 == 0) {
            global_size_2[0] *= NUM_WI_R_1;
            local_size_2[0] *= NUM_WI_R_1;
        } else if (OCL_DIM_R_1 == 1) {
            global_size_2[1] *= NUM_WI_R_1;
            local_size_2[1] *= NUM_WI_R_1;
        } else {
            global_size_2[2] *= NUM_WI_R_1;
            local_size_2[2] *= NUM_WI_R_1;
        }

        std::cout << "Kernel 2 NDRange:" << std::endl;
        std::cout << "  Global: " << global_size_2[0] << " x " << global_size_2[1] << " x " << global_size_2[2] << std::endl;
        std::cout << "  Local:  " << local_size_2[0] << " x " << local_size_2[1] << " x " << local_size_2[2] << std::endl;

        // Execute kernel_2
        cl_event event2;
        err = clEnqueueNDRangeKernel(queue, kernel2, 3, nullptr, global_size_2, local_size_2, 0, nullptr, &event2);
        if (err != CL_SUCCESS) {
            std::cerr << "Error enqueuing kernel tc_2: " << err << std::endl;
            return EXIT_FAILURE;
        }
        clWaitForEvents(1, &event2);
        
        cl_ulong start_time_2, end_time_2;
        clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time_2, nullptr);
        clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time_2, nullptr);
        std::cout << "✓ tc_2 executed in " << ((end_time_2 - start_time_2) / 1000000.0) << " ms" << std::endl;
        
        // Read result from buf_c
        clEnqueueReadBuffer(queue, buf_c, CL_TRUE, 0, c.size() * sizeof(float), c.data(), 0, nullptr, nullptr);
        
        clReleaseEvent(event2);
        clReleaseKernel(kernel2);
        clReleaseProgram(program2);
    } else {
        // No reduction needed, result is directly in res_g
        std::cout << "\nNo reduction needed (NUM_WG_R_1 = 1), reading directly from res_g" << std::endl;
        clEnqueueReadBuffer(queue, buf_res_g, CL_TRUE, 0, c.size() * sizeof(float), c.data(), 0, nullptr, nullptr);
    }


    // Compute CPU reference result
    std::cout << "\n=== Computing reference result on CPU ===" << std::endl;
    std::vector<float> ref_c(c_total);
    compute_reference(a, b, ref_c, M1, M2, M3, M4, M5, M6, N1);

    // Validate results
    std::cout << "\n" << std::string(50, '=') << std::endl;
    validate_result(c, ref_c, c_total);
    std::cout << std::string(50, '=') << std::endl;

    // Cleanup
    clReleaseEvent(event1);
    clReleaseKernel(kernel1);
    clReleaseProgram(program1);
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_c);
    clReleaseMemObject(buf_res_g);
    clReleaseMemObject(buf_int_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}
