#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <random>
#include <cmath>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "gemm_512x512x512_000005.cl"  // Tuning parameters

std::string read_kernel_source(const char *filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// CPU reference: Matrix Multiplication C = A * B with intermediate result reduction
// This mimics ATF's two-kernel pattern where intermediate results are accumulated
void compute_reference(const std::vector<float> &a, const std::vector<float> &b, 
                       std::vector<float> &c,
                       size_t M, size_t N, size_t K) {
    // C[M x N] = A[M x K] * B[K x N]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// Reduce intermediate results (used for multi-workgroup reductions)
void reduce_intermediate_results(std::vector<float> &c, const std::vector<float> &int_res,
                                 size_t M, size_t N) {
    // Add accumulated intermediate results to final result
    for (size_t i = 0; i < M * N; ++i) {
        c[i] += int_res[i];
    }
}

// Compare GPU result with reference (with floating-point tolerance)
bool validate_result(const std::vector<float> &gpu_out, const std::vector<float> &ref_out,
                     size_t size) {
    int mismatch_count = 0;
    float max_diff = 0.0f;
    const float tolerance = 0.1f;  // Allow reasonable floating-point errors for GPU computation
    
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
    const size_t M = 512, N = 512, K = 512;
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
    std::uniform_real_distribution<> dis(1.0, 10.0);
    
    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    std::vector<float> c(M * N, 0.0f);
    
    // Calculate intermediate result buffer size based on configuration
    // Size depends on destination levels and NUM_WG_R_1
    size_t int_res_size = M * N * NUM_WG_R_1;
    std::vector<float> int_res(int_res_size, 0.0f);  // Intermediate results buffer
    
    // Calculate res_g buffer size (used as 3rd arg to gemm_1)
    // Based on ATF's res_g_size calculation for kernel 1
    size_t res_g_size = M * N;
    if (G_CB_RES_DEST_LEVEL == 2) {
        res_g_size *= NUM_WG_R_1;
    }
    if (L_CB_RES_DEST_LEVEL == 2) {
        res_g_size *= NUM_WI_R_1;
    }
    std::vector<float> res_g(res_g_size, 0.0f);
    
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(dis(gen));
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(dis(gen));

    cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   a.size() * sizeof(float), a.data(), &err);
    cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   b.size() * sizeof(float), b.data(), &err);
    cl_mem buf_c = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   c.size() * sizeof(float), c.data(), &err);
    cl_mem buf_res_g = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       res_g.size() * sizeof(float), res_g.data(), &err);
    cl_mem buf_int_res = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         int_res.size() * sizeof(float), int_res.data(), &err);

    // Load and compile kernels
    std::string kernel_source_1 = read_kernel_source("./gemm_1.cl");
    std::string kernel_source_2 = read_kernel_source("./gemm_2.cl");
    const char *src_1 = kernel_source_1.c_str();
    const char *src_2 = kernel_source_2.c_str();
    const char *sources[] = {src_1, src_2};
    cl_program program = clCreateProgramWithSource(context, 2, sources, nullptr, &err);

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
        " -DINPUT_SIZE_R_1=" + std::to_string(INPUT_SIZE_R_1) +
        " -DL_CB_SIZE_R_1=" + std::to_string(L_CB_SIZE_R_1) +
        " -DP_CB_SIZE_R_1=" + std::to_string(P_CB_SIZE_R_1) +
        " -DOCL_DIM_R_1=" + std::to_string(OCL_DIM_R_1) +
        " -DNUM_WG_R_1=" + std::to_string(NUM_WG_R_1) +
        " -DNUM_WI_R_1=" + std::to_string(NUM_WI_R_1) +
        " -DL_REDUCTION=" + std::to_string(L_REDUCTION) +
        " -DP_WRITE_BACK=" + std::to_string(P_WRITE_BACK) +
        " -DL_WRITE_BACK=" + std::to_string(L_WRITE_BACK);

    err = clBuildProgram(program, 1, &device, compile_flags.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Compilation error:\n" << log.data() << std::endl;
        return EXIT_FAILURE;
    }

    // Create kernels
    cl_kernel kernel_1 = clCreateKernel(program, "gemm_1", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel gemm_1: " << err << std::endl;
        return EXIT_FAILURE;
    }

    cl_kernel kernel_2 = clCreateKernel(program, "gemm_2", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel gemm_2: " << err << std::endl;
        return EXIT_FAILURE;
    }

    // Set kernel_1 arguments (a, b, res_g, int_res)
    clSetKernelArg(kernel_1, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel_1, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel_1, 2, sizeof(cl_mem), &buf_res_g);
    clSetKernelArg(kernel_1, 3, sizeof(cl_mem), &buf_int_res);

    // Calculate global and local sizes
    size_t global_size[3], local_size[3];
    global_size[0] = ((OCL_DIM_L_1 == 0) * NUM_WG_L_1 * NUM_WI_L_1 + 
                      (OCL_DIM_L_2 == 0) * NUM_WG_L_2 * NUM_WI_L_2 +
                      (OCL_DIM_R_1 == 0) * NUM_WG_R_1 * NUM_WI_R_1);
    global_size[1] = ((OCL_DIM_L_1 == 1) * NUM_WG_L_1 * NUM_WI_L_1 + 
                      (OCL_DIM_L_2 == 1) * NUM_WG_L_2 * NUM_WI_L_2 +
                      (OCL_DIM_R_1 == 1) * NUM_WG_R_1 * NUM_WI_R_1);
    global_size[2] = ((OCL_DIM_L_1 == 2) * NUM_WG_L_1 * NUM_WI_L_1 + 
                      (OCL_DIM_L_2 == 2) * NUM_WG_L_2 * NUM_WI_L_2 +
                      (OCL_DIM_R_1 == 2) * NUM_WG_R_1 * NUM_WI_R_1);
    
    local_size[0] = ((OCL_DIM_L_1 == 0) * NUM_WI_L_1 + 
                     (OCL_DIM_L_2 == 0) * NUM_WI_L_2 +
                     (OCL_DIM_R_1 == 0) * NUM_WI_R_1);
    local_size[1] = ((OCL_DIM_L_1 == 1) * NUM_WI_L_1 + 
                     (OCL_DIM_L_2 == 1) * NUM_WI_L_2 +
                     (OCL_DIM_R_1 == 1) * NUM_WI_R_1);
    local_size[2] = ((OCL_DIM_L_1 == 2) * NUM_WI_L_1 + 
                     (OCL_DIM_L_2 == 2) * NUM_WI_L_2 +
                     (OCL_DIM_R_1 == 2) * NUM_WI_R_1);

    // Ensure local sizes are at least 1
    for (int i = 0; i < 3; ++i) {
        if (local_size[i] == 0) local_size[i] = 1;
    }

    std::cout << "Global size: " << global_size[0] << " x " << global_size[1] << " x " << global_size[2] << std::endl;
    std::cout << "Local size: " << local_size[0] << " x " << local_size[1] << " x " << local_size[2] << std::endl;

    // Execute gemm_1 kernel
    std::cout << "\n--- Executing gemm_1 kernel ---" << std::endl;
    cl_event event_1;
    err = clEnqueueNDRangeKernel(queue, kernel_1, 3, nullptr, global_size, local_size, 0, nullptr, &event_1);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing gemm_1 kernel: " << err << std::endl;
        return EXIT_FAILURE;
    }

    clWaitForEvents(1, &event_1);

    // Check if we need gemm_2 (reduction kernel) - needed when NUM_WG_R_1 > 1
    bool needs_reduction = (NUM_WG_R_1 > 1);
    
    if (needs_reduction) {
        // Set kernel_2 arguments (int_res, res_g, c)
        clSetKernelArg(kernel_2, 0, sizeof(cl_mem), &buf_int_res);
        clSetKernelArg(kernel_2, 1, sizeof(cl_mem), &buf_res_g);
        clSetKernelArg(kernel_2, 2, sizeof(cl_mem), &buf_c);

        // Calculate global and local sizes for gemm_2 (reduction kernel)
        size_t global_size_2[3], local_size_2[3];
        global_size_2[0] = ((OCL_DIM_L_1 == 0) * NUM_WG_L_1 * NUM_WI_L_1 + 
                            (OCL_DIM_L_2 == 0) * NUM_WG_L_2 * NUM_WI_L_2 +
                            (OCL_DIM_R_1 == 0) * NUM_WI_R_1);
        global_size_2[1] = ((OCL_DIM_L_1 == 1) * NUM_WG_L_1 * NUM_WI_L_1 + 
                            (OCL_DIM_L_2 == 1) * NUM_WG_L_2 * NUM_WI_L_2 +
                            (OCL_DIM_R_1 == 1) * NUM_WI_R_1);
        global_size_2[2] = ((OCL_DIM_L_1 == 2) * NUM_WG_L_1 * NUM_WI_L_1 + 
                            (OCL_DIM_L_2 == 2) * NUM_WG_L_2 * NUM_WI_L_2 +
                            (OCL_DIM_R_1 == 2) * NUM_WI_R_1);
        
        local_size_2[0] = ((OCL_DIM_L_1 == 0) * NUM_WI_L_1 + 
                           (OCL_DIM_L_2 == 0) * NUM_WI_L_2 +
                           (OCL_DIM_R_1 == 0) * NUM_WI_R_1);
        local_size_2[1] = ((OCL_DIM_L_1 == 1) * NUM_WI_L_1 + 
                           (OCL_DIM_L_2 == 1) * NUM_WI_L_2 +
                           (OCL_DIM_R_1 == 1) * NUM_WI_R_1);
        local_size_2[2] = ((OCL_DIM_L_1 == 2) * NUM_WI_L_1 + 
                           (OCL_DIM_L_2 == 2) * NUM_WI_L_2 +
                           (OCL_DIM_R_1 == 2) * NUM_WI_R_1);

        // Ensure local sizes are at least 1
        for (int i = 0; i < 3; ++i) {
            if (local_size_2[i] == 0) local_size_2[i] = 1;
        }

        std::cout << "\n--- Executing gemm_2 (reduction) kernel ---" << std::endl;
        std::cout << "Global size: " << global_size_2[0] << " x " << global_size_2[1] << " x " << global_size_2[2] << std::endl;
        std::cout << "Local size: " << local_size_2[0] << " x " << local_size_2[1] << " x " << local_size_2[2] << std::endl;

        cl_event event_2;
        err = clEnqueueNDRangeKernel(queue, kernel_2, 3, nullptr, global_size_2, local_size_2, 0, nullptr, &event_2);
        if (err != CL_SUCCESS) {
            std::cerr << "Error enqueuing gemm_2 kernel: " << err << std::endl;
            return EXIT_FAILURE;
        }

        clWaitForEvents(1, &event_2);

        // Get kernel_2 execution time
        cl_ulong start_time_2, end_time_2;
        clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time_2, nullptr);
        clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time_2, nullptr);
        unsigned long long runtime_ns_2 = end_time_2 - start_time_2;

        std::cout << "GPU gemm_2 kernel execution time: " << (runtime_ns_2 / 1000000.0) << " ms" << std::endl;
        clReleaseEvent(event_2);
    } else {
        std::cout << "No reduction needed (NUM_WG_R_1 = 1)" << std::endl;
    }

    // Read back GPU results
    if (needs_reduction) {
        // If reduction was performed, result is in buf_c
        clEnqueueReadBuffer(queue, buf_c, CL_TRUE, 0, c.size() * sizeof(float), c.data(), 0, nullptr, nullptr);
    } else {
        // If no reduction, result is in buf_res_g (should be same size as c when NUM_WG_R_1 = 1)
        clEnqueueReadBuffer(queue, buf_res_g, CL_TRUE, 0, c.size() * sizeof(float), c.data(), 0, nullptr, nullptr);
    }

    // Compute CPU reference result
    std::vector<float> ref_c(M * N);
    std::cout << "Computing reference result on CPU..." << std::endl;
    compute_reference(a, b, ref_c, M, N, K);

    // Get gemm_1 kernel execution time
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event_1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
    clGetEventProfilingInfo(event_1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
    unsigned long long runtime_ns = end_time - start_time;

    std::cout << "GPU gemm_1 kernel execution time: " << (runtime_ns / 1000000.0) << " ms" << std::endl;

    // Validate results
    std::cout << "\n" << std::string(50, '=') << std::endl;
    validate_result(c, ref_c, M * N);
    std::cout << std::string(50, '=') << std::endl;

    // Cleanup
    clReleaseEvent(event_1);
    clReleaseKernel(kernel_1);
    clReleaseKernel(kernel_2);
    clReleaseProgram(program);
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_c);
    clReleaseMemObject(buf_res_g);
    clReleaseMemObject(buf_int_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}
