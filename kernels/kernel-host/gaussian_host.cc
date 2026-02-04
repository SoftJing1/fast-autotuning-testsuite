#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <random>
#include <filesystem>
#include <nlohmann/json.hpp>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

using json = nlohmann::json;

// Tuning parameters and input size struct
struct GaussianConfig {
    // Input dimensions
    int input_size_h = 1024;
    int input_size_w = 1024;
    
    // Tuning parameters with default values
    int g_cb_res_dest_level = 2;
    int l_cb_res_dest_level = 0;
    int p_cb_res_dest_level = 0;
    int images_cache_lcl = 0;
    int images_cache_prv = 0;
    int filter_cache_lcl = 0;
    int filter_cache_prv = 0;
    int out_cache_prv = 0;
    int wg_1_ocl_dim = 0;
    int wg_2_ocl_dim = 1;
    int wi_1_ocl_dim = 0;
    int wi_2_ocl_dim = 1;
    int glb_1 = 1;
    int wg_1 = 1;
    int lcl_1 = 1;
    int wi_1 = 1;
    int prv_1 = 1;
    int glb_2 = 1;
    int wg_2 = 1;
    int lcl_2 = 1;
    int wi_2 = 1;
    int prv_2 = 1;
};

// Load configuration from JSON file
GaussianConfig load_config_from_json(const std::string& json_path) {
    GaussianConfig config;
    
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file: " << json_path << std::endl;
        exit(EXIT_FAILURE);
    }
    
    try {
        json j;
        file >> j;
        
        // Load input sizes
        if (j.contains("input_size_h")) config.input_size_h = j["input_size_h"];
        if (j.contains("input_size_w")) config.input_size_w = j["input_size_w"];
        
        // Load tuning parameters
        if (j.contains("g_cb_res_dest_level")) config.g_cb_res_dest_level = j["g_cb_res_dest_level"];
        if (j.contains("l_cb_res_dest_level")) config.l_cb_res_dest_level = j["l_cb_res_dest_level"];
        if (j.contains("p_cb_res_dest_level")) config.p_cb_res_dest_level = j["p_cb_res_dest_level"];
        if (j.contains("images_cache_lcl")) config.images_cache_lcl = j["images_cache_lcl"];
        if (j.contains("images_cache_prv")) config.images_cache_prv = j["images_cache_prv"];
        if (j.contains("filter_cache_lcl")) config.filter_cache_lcl = j["filter_cache_lcl"];
        if (j.contains("filter_cache_prv")) config.filter_cache_prv = j["filter_cache_prv"];
        if (j.contains("out_cache_prv")) config.out_cache_prv = j["out_cache_prv"];
        if (j.contains("wg_1_ocl_dim")) config.wg_1_ocl_dim = j["wg_1_ocl_dim"];
        if (j.contains("wg_2_ocl_dim")) config.wg_2_ocl_dim = j["wg_2_ocl_dim"];
        if (j.contains("wi_1_ocl_dim")) config.wi_1_ocl_dim = j["wi_1_ocl_dim"];
        if (j.contains("wi_2_ocl_dim")) config.wi_2_ocl_dim = j["wi_2_ocl_dim"];
        if (j.contains("glb_1")) config.glb_1 = j["glb_1"];
        if (j.contains("wg_1")) config.wg_1 = j["wg_1"];
        if (j.contains("lcl_1")) config.lcl_1 = j["lcl_1"];
        if (j.contains("wi_1")) config.wi_1 = j["wi_1"];
        if (j.contains("prv_1")) config.prv_1 = j["prv_1"];
        if (j.contains("glb_2")) config.glb_2 = j["glb_2"];
        if (j.contains("wg_2")) config.wg_2 = j["wg_2"];
        if (j.contains("lcl_2")) config.lcl_2 = j["lcl_2"];
        if (j.contains("wi_2")) config.wi_2 = j["wi_2"];
        if (j.contains("prv_2")) config.prv_2 = j["prv_2"];
        
    } catch (const json::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return config;
}

std::string get_kernel_template_path(const char* kernel_name = "gaussian_static_1.cl") {
    namespace fs = std::filesystem;
    
    #ifndef KERNEL_TEMPLATE_DIR
        std::cerr << "Error: KERNEL_TEMPLATE_DIR not defined. Please build with CMake." << std::endl;
        exit(EXIT_FAILURE);
    #endif
    
    fs::path template_path = fs::path(KERNEL_TEMPLATE_DIR) / kernel_name;
    if (!fs::exists(template_path)) {
        std::cerr << "Error: Kernel template not found at: " << template_path.string() << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return fs::absolute(template_path).string();
}

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

int main(int argc, char *argv[]) {
    // Load configuration from JSON file
    std::string config_path = "config.json";
    if (argc > 1) {
        config_path = argv[1];
    }
    else{
        std::cout << "No config file provided. Please provide a configuration JSON file as an argument." << std::endl;
        return EXIT_FAILURE;
    }
    
    GaussianConfig config = load_config_from_json(config_path);
    
    std::cout << "Using configuration from: " << config_path << std::endl;
    std::cout << "Input size: " << config.input_size_h << " x " << config.input_size_w << std::endl;
    std::cout << "Tuning parameters:" << std::endl;
    std::cout << "  G_CB_RES_DEST_LEVEL: " << config.g_cb_res_dest_level << std::endl;
    std::cout << "  L_CB_RES_DEST_LEVEL: " << config.l_cb_res_dest_level << std::endl;
    std::cout << "  WG_1: " << config.wg_1 << ", WI_1: " << config.wi_1 << std::endl;
    std::cout << "  WG_2: " << config.wg_2 << ", WI_2: " << config.wi_2 << std::endl;
    
    const size_t H = config.input_size_h;
    const size_t W = config.input_size_w;
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

    // Load kernel from template using relative path
    std::string kernel_template_path = get_kernel_template_path("gaussian_static_1.cl");
    std::string kernel_source = read_kernel_source(kernel_template_path.c_str());
    const char *src = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);

    std::string compile_flags = "-DTYPE_T=float -DTYPE_TS=float "
        "-DG_CB_RES_DEST_LEVEL=" + std::to_string(config.g_cb_res_dest_level) +
        " -DL_CB_RES_DEST_LEVEL=" + std::to_string(config.l_cb_res_dest_level) +
        " -DP_CB_RES_DEST_LEVEL=" + std::to_string(config.p_cb_res_dest_level) +
        " -DIMAGES_CACHE_LCL=" + std::to_string(config.images_cache_lcl) +
        " -DIMAGES_CACHE_PRV=" + std::to_string(config.images_cache_prv) +
        " -DFILTER_CACHE_LCL=" + std::to_string(config.filter_cache_lcl) +
        " -DFILTER_CACHE_PRV=" + std::to_string(config.filter_cache_prv) +
        " -DOUT_CACHE_PRV=" + std::to_string(config.out_cache_prv) +
        " -DWG_1_OCL_DIM=" + std::to_string(config.wg_1_ocl_dim) +
        " -DWG_2_OCL_DIM=" + std::to_string(config.wg_2_ocl_dim) +
        " -DWI_1_OCL_DIM=" + std::to_string(config.wi_1_ocl_dim) +
        " -DWI_2_OCL_DIM=" + std::to_string(config.wi_2_ocl_dim) +
        " -DGLB_1=" + std::to_string(config.glb_1) +
        " -DWG_1=" + std::to_string(config.wg_1) +
        " -DLCL_1=" + std::to_string(config.lcl_1) +
        " -DWI_1=" + std::to_string(config.wi_1) +
        " -DPRV_1=" + std::to_string(config.prv_1) +
        " -DGLB_2=" + std::to_string(config.glb_2) +
        " -DWG_2=" + std::to_string(config.wg_2) +
        " -DLCL_2=" + std::to_string(config.lcl_2) +
        " -DWI_2=" + std::to_string(config.wi_2) +
        " -DPRV_2=" + std::to_string(config.prv_2) +
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
    global_size[0] = ((config.wg_1_ocl_dim == 0) * config.glb_1 + (config.wg_2_ocl_dim == 0) * config.glb_2) * 
                     ((config.wi_1_ocl_dim == 0) * config.wi_1 + (config.wi_2_ocl_dim == 0) * config.wi_2);
    global_size[1] = ((config.wg_1_ocl_dim == 1) * config.glb_1 + (config.wg_2_ocl_dim == 1) * config.glb_2) * 
                     ((config.wi_1_ocl_dim == 1) * config.wi_1 + (config.wi_2_ocl_dim == 1) * config.wi_2);
    local_size[0] = (config.wi_1_ocl_dim == 0) * config.wi_1 + (config.wi_2_ocl_dim == 0) * config.wi_2;
    local_size[1] = (config.wi_1_ocl_dim == 1) * config.wi_1 + (config.wi_2_ocl_dim == 1) * config.wi_2;

    // Ensure local sizes are at least 1
    if (local_size[0] == 0 || local_size[1] == 0) {
        std::cerr << "Error: Local size dimensions cannot be zero. Wrong configuration" << std::endl;
        return EXIT_FAILURE;
    }

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
    bool valid = validate_result(out, ref_out, H * W);
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

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
