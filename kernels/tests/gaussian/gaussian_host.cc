#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <random>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "gaussian_1024x1024_000000.cl"  // Tuning parameters

std::string read_kernel_source(const char *filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// CPU reference: 5x5 filter from gaussian_static_1.cl kernel
void compute_reference(const std::vector<float> &in_padded, std::vector<float> &out,
                       size_t H, size_t W) {
    // 5x5 filter kernel from the OpenCL code
    // Format: kernel[dy][dx] where (0,0) is top-left corner
    float kernel[5][5] = {
        {2.0f,  4.0f,  5.0f,  4.0f,  2.0f},
        {4.0f,  9.0f, 12.0f,  9.0f,  4.0f},
        {5.0f, 12.0f, 15.0f, 12.0f,  5.0f},
        {4.0f,  9.0f, 12.0f,  9.0f,  4.0f},
        {2.0f,  4.0f,  5.0f,  4.0f,  2.0f}
    };

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            float result = 0.0f;
            
            // Apply 5x5 convolution
            for (int dy = 0; dy < 5; ++dy) {
                for (int dx = 0; dx < 5; ++dx) {
                    size_t iy = y + dy;        // Input is padded, so y+dy gives correct position
                    size_t ix = x + dx;
                    result += in_padded[iy * (W + 4) + ix] * kernel[dy][dx];
                }
            }
            
            out[y * W + x] = result;
        }
    }
}

// Compare GPU result with reference (with floating-point tolerance)
bool validate_result(const std::vector<float> &gpu_out, const std::vector<float> &ref_out,
                     size_t size) {
    int mismatch_count = 0;
    float max_diff = 0.0f;
    const float tolerance = 1e-2f;  // Allow small floating-point errors
    
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
    const size_t H = 1024, W = 1024;
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
    std::uniform_real_distribution<> dis(10.0, 100.0);
    
    std::vector<float> in((H + 4) * (W + 4));
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(dis(gen));
    
    std::vector<float> out(H * W, 0);
    std::vector<float> dummy(H * W, 0);  // Dummy buffer (unused in kernel)

    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    in.size() * sizeof(float), in.data(), &err);
    cl_mem buf_dummy = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       dummy.size() * sizeof(float), nullptr, &err);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     out.size() * sizeof(float), out.data(), &err);

    // Load and compile kernel
    std::string kernel_source = read_kernel_source("./gaussian_static_1.cl");
    const char *src = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);

    std::string compile_flags = "-DTYPE_T=float -DTYPE_TS=float "
        "-DG_CB_RES_DEST_LEVEL=" + std::to_string(G_CB_RES_DEST_LEVEL) +
        " -DL_CB_RES_DEST_LEVEL=" + std::to_string(L_CB_RES_DEST_LEVEL) +
        " -DP_CB_RES_DEST_LEVEL=" + std::to_string(P_CB_RES_DEST_LEVEL) +
        " -DIMAGES_CACHE_LCL=" + std::to_string(IMAGES_CACHE_LCL) +
        " -DIMAGES_CACHE_PRV=" + std::to_string(IMAGES_CACHE_PRV) +
        " -DFILTER_CACHE_LCL=" + std::to_string(FILTER_CACHE_LCL) +
        " -DFILTER_CACHE_PRV=" + std::to_string(FILTER_CACHE_PRV) +
        " -DOUT_CACHE_PRV=" + std::to_string(OUT_CACHE_PRV) +
        " -DWG_1_OCL_DIM=" + std::to_string(WG_1_OCL_DIM) +
        " -DWG_2_OCL_DIM=" + std::to_string(WG_2_OCL_DIM) +
        " -DWI_1_OCL_DIM=" + std::to_string(WI_1_OCL_DIM) +
        " -DWI_2_OCL_DIM=" + std::to_string(WI_2_OCL_DIM) +
        " -DGLB_1=" + std::to_string(GLB_1) +
        " -DWG_1=" + std::to_string(WG_1) +
        " -DLCL_1=" + std::to_string(LCL_1) +
        " -DWI_1=" + std::to_string(WI_1) +
        " -DPRV_1=" + std::to_string(PRV_1) +
        " -DGLB_2=" + std::to_string(GLB_2) +
        " -DWG_2=" + std::to_string(WG_2) +
        " -DLCL_2=" + std::to_string(LCL_2) +
        " -DWI_2=" + std::to_string(WI_2) +
        " -DPRV_2=" + std::to_string(PRV_2) +
        " -DINPUT_SIZE_1=" + std::to_string(H) +
        " -DINPUT_SIZE_2=" + std::to_string(W);

    err = clBuildProgram(program, 1, &device, compile_flags.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Compilation error:\n" << log.data() << std::endl;
        return EXIT_FAILURE;
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "gaussian_1", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel: " << err << std::endl;
        return EXIT_FAILURE;
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dummy);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);

    // Calculate global and local sizes
    size_t global_size[2], local_size[2];
    global_size[0] = ((WG_1_OCL_DIM == 0) * GLB_1 + (WG_2_OCL_DIM == 0) * GLB_2) * 
                     ((WI_1_OCL_DIM == 0) * WI_1 + (WI_2_OCL_DIM == 0) * WI_2);
    global_size[1] = ((WG_1_OCL_DIM == 1) * GLB_1 + (WG_2_OCL_DIM == 1) * GLB_2) * 
                     ((WI_1_OCL_DIM == 1) * WI_1 + (WI_2_OCL_DIM == 1) * WI_2);
    local_size[0] = (WI_1_OCL_DIM == 0) * WI_1 + (WI_2_OCL_DIM == 0) * WI_2;
    local_size[1] = (WI_1_OCL_DIM == 1) * WI_1 + (WI_2_OCL_DIM == 1) * WI_2;

    // Ensure local sizes are at least 1
    if (local_size[0] == 0) local_size[0] = 1;
    if (local_size[1] == 0) local_size[1] = 1;

    std::cout << "Global size: " << global_size[0] << " x " << global_size[1] << std::endl;
    std::cout << "Local size: " << local_size[0] << " x " << local_size[1] << std::endl;

    // Execute kernel
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel: " << err << std::endl;
        return EXIT_FAILURE;
    }

    clWaitForEvents(1, &event);

    // Read back GPU result
    clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, out.size() * sizeof(float), out.data(), 0, nullptr, nullptr);

    // Compute CPU reference result
    std::vector<float> ref_out(H * W);
    std::cout << "Computing reference result on CPU..." << std::endl;
    compute_reference(in, ref_out, H, W);

    // Get kernel execution time
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
    unsigned long long runtime_ns = end_time - start_time;

    std::cout << "GPU kernel execution time: " << (runtime_ns / 1000000.0) << " ms" << std::endl;

    // Validate results
    std::cout << "\n" << std::string(50, '=') << std::endl;
    validate_result(out, ref_out, H * W);
    std::cout << std::string(50, '=') << std::endl;

    // Cleanup
    clReleaseEvent(event);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_dummy);
    clReleaseMemObject(buf_out);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}
