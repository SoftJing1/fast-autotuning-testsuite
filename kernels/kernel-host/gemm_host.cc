#include <CL/cl_platform.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

using json = nlohmann::json;

const char *get_opencl_error_string(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  default:
    return "UNKNOWN_ERROR (vendor-specific or invalid error code)";
  }
}

// GEMM configuration struct with tuning parameters and input sizes
struct GemmConfig {
  // Input dimensions
  int M = 512;
  int N = 512;
  int K = 512;

  // Tuning parameters with default values
  int cache_l_cb = 0;
  int cache_p_cb = 0;
  int g_cb_res_dest_level = 0;
  int l_cb_res_dest_level = 0;
  int p_cb_res_dest_level = 0;
  int input_size_l_1 = 512;
  int l_cb_size_l_1 = 1;
  int p_cb_size_l_1 = 1;
  int ocl_dim_l_1 = 0;
  int num_wg_l_1 = 1;
  int num_wi_l_1 = 1;
  int input_size_l_2 = 512;
  int l_cb_size_l_2 = 1;
  int p_cb_size_l_2 = 1;
  int ocl_dim_l_2 = 0;
  int num_wg_l_2 = 1;
  int num_wi_l_2 = 1;
  int input_size_r_1 = 512;
  int l_cb_size_r_1 = 1;
  int p_cb_size_r_1 = 1;
  int ocl_dim_r_1 = 0;
  int num_wg_r_1 = 1;
  int num_wi_r_1 = 1;
  int l_reduction = 0;
  int p_write_back = 0;
  int l_write_back = 0;
};

// Load configuration from JSON file
GemmConfig load_config_from_json(const std::string &json_path) {
  GemmConfig config;

  std::ifstream file(json_path);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open config file: " << json_path
              << std::endl;
    std::cerr << "Using default parameters." << std::endl;
    return config;
  }

  try {
    json j;
    file >> j;

    // Load input sizes
    if (j.contains("M"))
      config.M = j["M"];
    if (j.contains("N"))
      config.N = j["N"];
    if (j.contains("K"))
      config.K = j["K"];

    // Load tuning parameters
    if (j.contains("cache_l_cb"))
      config.cache_l_cb = j["cache_l_cb"];
    if (j.contains("cache_p_cb"))
      config.cache_p_cb = j["cache_p_cb"];
    if (j.contains("g_cb_res_dest_level"))
      config.g_cb_res_dest_level = j["g_cb_res_dest_level"];
    if (j.contains("l_cb_res_dest_level"))
      config.l_cb_res_dest_level = j["l_cb_res_dest_level"];
    if (j.contains("p_cb_res_dest_level"))
      config.p_cb_res_dest_level = j["p_cb_res_dest_level"];
    if (j.contains("input_size_l_1"))
      config.input_size_l_1 = j["input_size_l_1"];
    if (j.contains("l_cb_size_l_1"))
      config.l_cb_size_l_1 = j["l_cb_size_l_1"];
    if (j.contains("p_cb_size_l_1"))
      config.p_cb_size_l_1 = j["p_cb_size_l_1"];
    if (j.contains("ocl_dim_l_1"))
      config.ocl_dim_l_1 = j["ocl_dim_l_1"];
    if (j.contains("num_wg_l_1"))
      config.num_wg_l_1 = j["num_wg_l_1"];
    if (j.contains("num_wi_l_1"))
      config.num_wi_l_1 = j["num_wi_l_1"];
    if (j.contains("input_size_l_2"))
      config.input_size_l_2 = j["input_size_l_2"];
    if (j.contains("l_cb_size_l_2"))
      config.l_cb_size_l_2 = j["l_cb_size_l_2"];
    if (j.contains("p_cb_size_l_2"))
      config.p_cb_size_l_2 = j["p_cb_size_l_2"];
    if (j.contains("ocl_dim_l_2"))
      config.ocl_dim_l_2 = j["ocl_dim_l_2"];
    if (j.contains("num_wg_l_2"))
      config.num_wg_l_2 = j["num_wg_l_2"];
    if (j.contains("num_wi_l_2"))
      config.num_wi_l_2 = j["num_wi_l_2"];
    if (j.contains("input_size_r_1"))
      config.input_size_r_1 = j["input_size_r_1"];
    if (j.contains("l_cb_size_r_1"))
      config.l_cb_size_r_1 = j["l_cb_size_r_1"];
    if (j.contains("p_cb_size_r_1"))
      config.p_cb_size_r_1 = j["p_cb_size_r_1"];
    if (j.contains("ocl_dim_r_1"))
      config.ocl_dim_r_1 = j["ocl_dim_r_1"];
    if (j.contains("num_wg_r_1"))
      config.num_wg_r_1 = j["num_wg_r_1"];
    if (j.contains("num_wi_r_1"))
      config.num_wi_r_1 = j["num_wi_r_1"];
    if (j.contains("l_reduction"))
      config.l_reduction = j["l_reduction"];
    if (j.contains("p_write_back"))
      config.p_write_back = j["p_write_back"];
    if (j.contains("l_write_back"))
      config.l_write_back = j["l_write_back"];

  } catch (const json::exception &e) {
    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    std::cerr << "Using default parameters." << std::endl;
  }

  return config;
}

std::string get_kernel_template_path(const char *kernel_name) {
  namespace fs = std::filesystem;

#ifndef KERNEL_TEMPLATE_DIR
  std::cerr
      << "Error: KERNEL_TEMPLATE_DIR not defined. Please build with CMake."
      << std::endl;
  exit(EXIT_FAILURE);
#endif

  fs::path template_path = fs::path(KERNEL_TEMPLATE_DIR) / kernel_name;
  if (!fs::exists(template_path)) {
    std::cerr << "Error: Kernel template not found at: "
              << template_path.string() << std::endl;
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
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

// CPU reference: Matrix Multiplication C = A * B with intermediate result
// reduction This mimics ATF's two-kernel pattern where intermediate results are
// accumulated
void compute_reference(const std::vector<float> &a, const std::vector<float> &b,
                       std::vector<float> &c, size_t M, size_t N, size_t K) {
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
void reduce_intermediate_results(std::vector<float> &c,
                                 const std::vector<float> &int_res, size_t M,
                                 size_t N) {
  // Add accumulated intermediate results to final result
  for (size_t i = 0; i < M * N; ++i) {
    c[i] += int_res[i];
  }
}

// Compare Device result with reference (with floating-point tolerance)
bool validate_result(const std::vector<float> &device_out,
                     const std::vector<float> &ref_out, size_t size) {
  int mismatch_count = 0;
  float max_diff = 0.0f;
  const float tolerance =
      0.1f; // Allow reasonable floating-point errors for Device computation

  std::cout << "Validating computation..." << std::endl;

  for (size_t i = 0; i < size; ++i) {
    float diff = std::abs(device_out[i] - ref_out[i]);
    max_diff = std::max(max_diff, diff);

    if (diff > tolerance) {
      mismatch_count++;
      if (mismatch_count <= 5) { // Print first 5 mismatches
        std::cout << "  Mismatch at index " << i << std::endl;
        std::cout << "    Reference: " << ref_out[i] << std::endl;
        std::cout << "    Device:    " << device_out[i] << std::endl;
        std::cout << "    Diff:      " << diff << std::endl;
      }
    }
  }

  if (mismatch_count == 0) {
    std::cout << "✓ Result is CORRECT (max diff: " << max_diff << ")"
              << std::endl;
    return true;
  } else {
    std::cout << "✗ Result is INCORRECT" << std::endl;
    std::cout << "  Mismatches > " << tolerance << ": " << mismatch_count << "/"
              << size << std::endl;
    std::cout << "  Max difference: " << max_diff << std::endl;
    return false;
  }
}

bool select_cpu_platform_and_device(cl_platform_id &platform,
                                    cl_device_id &device) {
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    std::cerr << "Error querying OpenCL platforms: " << err << std::endl;
    return false;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Error getting platform list: " << err << std::endl;
    return false;
  }

  for (const auto current_platform : platforms) {
    cl_uint cpu_device_count = 0;
    cl_int device_query_err =
        clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_CPU, 0, nullptr,
                       &cpu_device_count);

    if (device_query_err == CL_DEVICE_NOT_FOUND || cpu_device_count == 0) {
      continue;
    }
    if (device_query_err != CL_SUCCESS) {
      continue;
    }

    err = clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_CPU, 1, &device,
                         nullptr);
    if (err == CL_SUCCESS) {
      platform = current_platform;
      return true;
    }
  }

  std::cerr << "Error: No OpenCL CPU device found on any platform." << std::endl;
  std::cerr << "Make sure a CPU OpenCL runtime (e.g., Intel OpenCL CPU) is "
               "installed."
            << std::endl;
  return false;
}

void dump_system_diagnostics(const std::string &reason) {
  std::cerr << "Collecting recent system diagnostics (" << reason << ")..."
            << std::endl;

  int rc = std::system(
      "journalctl -k -n 120 --no-pager 2>/dev/null | "
      "grep -Ei 'NVRM|Xid|nvidia|amdgpu|i915|xe|drm|opencl|ocl|gpu|hsa|rocm|"
      "fault|hang|reset|wedged|segfault' || true");
  if (rc == -1) {
    std::cerr << "Failed to invoke journalctl for diagnostics." << std::endl;
  }

  rc = std::system(
      "coredumpctl list --no-pager 2>/dev/null | "
      "grep -Ei 'gaussian|gemm' | tail -n 10 || true");
  if (rc == -1) {
    std::cerr << "Failed to invoke coredumpctl for diagnostics." << std::endl;
  }
}

bool is_all_zero_output(const std::vector<float> &values) {
  if (values.empty()) {
    return false;
  }

  for (float value : values) {
    if (value != 0.0f) {
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  using WallClock = std::chrono::steady_clock;
  const auto program_start_time = WallClock::now();
  auto phase_start_time = program_start_time;
  std::vector<std::pair<std::string, double>> phase_timings_ms;

  auto record_phase = [&](const std::string &phase_name) {
    const auto now = WallClock::now();
    const auto phase_duration_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            now - phase_start_time)
            .count();
    phase_timings_ms.emplace_back(phase_name, phase_duration_ms);
    phase_start_time = now;
  };

  // Load configuration from JSON file
  std::string config_path = "config.json";
  if (argc > 1) {
    config_path = argv[1];
  } else {
    std::cout << "No config file provided. Please provide a configuration JSON "
                 "file as an argument."
              << std::endl;
    return EXIT_FAILURE;
  }

  GemmConfig config = load_config_from_json(config_path);
  record_phase("Arg parsing + config load");

  std::cout << "Using configuration from: " << config_path << std::endl;
  std::cout << "Matrix dimensions: M=" << config.M << ", N=" << config.N
            << ", K=" << config.K << std::endl;

  const size_t M = config.M;
  const size_t N = config.N;
  const size_t K = config.K;
  cl_int err;

  // Get platform and CPU device
  cl_platform_id platform;
  cl_device_id device;
  if (!select_cpu_platform_and_device(platform, device)) {
    return EXIT_FAILURE;
  }

  char platform_name[256];
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                    platform_name, nullptr);
  char device_name[256];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name,
                  nullptr);
  std::cout << "Platform: " << platform_name << std::endl;
  std::cout << "Device:   " << device_name << std::endl;

  // Create context and command queue
  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating context: " << err << std::endl;
    return EXIT_FAILURE;
  }

  cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES,
                                       CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, properties, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating command queue: " << err << std::endl;
    return EXIT_FAILURE;
  }

  // Create buffers with deterministic integer-valued input data (stored as
  // float). This keeps values exactly representable and makes CPU/GPU
  // verification reproducible.

  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);

  // Calculate intermediate result buffer size based on configuration
  // Size depends on destination levels and NUM_WG_R_1
  size_t int_res_size = M * N * config.num_wg_r_1;
  std::vector<float> int_res(int_res_size, 0.0f); // Intermediate results buffer

  // Calculate res_g buffer size (used as 3rd arg to gemm_1)
  // Based on ATF's res_g_size calculation for kernel 1
  size_t res_g_size = M * N;
  if (config.g_cb_res_dest_level == 2) {
    res_g_size *= config.num_wg_r_1;
  }
  if (config.l_cb_res_dest_level == 2) {
    res_g_size *= config.num_wi_r_1;
  }
  std::vector<float> res_g(res_g_size, 0.0f);

  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>((i % 7) - 3); // [-3, 3]
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>((i % 5) - 2); // [-2, 2]

  cl_mem buf_a =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     a.size() * sizeof(float), a.data(), &err);
  cl_mem buf_b =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     b.size() * sizeof(float), b.data(), &err);
  cl_mem buf_c =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     c.size() * sizeof(float), c.data(), &err);
  cl_mem buf_res_g =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     res_g.size() * sizeof(float), res_g.data(), &err);
  cl_mem buf_int_res =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     int_res.size() * sizeof(float), int_res.data(), &err);

  // Load and compile kernels
  std::string kernel_source_1 =
      read_kernel_source(get_kernel_template_path("gemm_1.cl").c_str());
  std::string kernel_source_2 =
      read_kernel_source(get_kernel_template_path("gemm_2.cl").c_str());
  const char *src_1 = kernel_source_1.c_str();
  const char *src_2 = kernel_source_2.c_str();
  const char *sources[] = {src_1, src_2};
  cl_program program =
      clCreateProgramWithSource(context, 2, sources, nullptr, &err);

  std::string compile_flags =
      "-DTYPE_T=float -DTYPE_TS=float "
      "-DCACHE_L_CB=" +
      std::to_string(config.cache_l_cb) +
      " -DCACHE_P_CB=" + std::to_string(config.cache_p_cb) +
      " -DG_CB_RES_DEST_LEVEL=" + std::to_string(config.g_cb_res_dest_level) +
      " -DL_CB_RES_DEST_LEVEL=" + std::to_string(config.l_cb_res_dest_level) +
      " -DP_CB_RES_DEST_LEVEL=" + std::to_string(config.p_cb_res_dest_level) +
      " -DINPUT_SIZE_L_1=" + std::to_string(config.input_size_l_1) +
      " -DL_CB_SIZE_L_1=" + std::to_string(config.l_cb_size_l_1) +
      " -DP_CB_SIZE_L_1=" + std::to_string(config.p_cb_size_l_1) +
      " -DOCL_DIM_L_1=" + std::to_string(config.ocl_dim_l_1) +
      " -DNUM_WG_L_1=" + std::to_string(config.num_wg_l_1) +
      " -DNUM_WI_L_1=" + std::to_string(config.num_wi_l_1) +
      " -DINPUT_SIZE_L_2=" + std::to_string(config.input_size_l_2) +
      " -DL_CB_SIZE_L_2=" + std::to_string(config.l_cb_size_l_2) +
      " -DP_CB_SIZE_L_2=" + std::to_string(config.p_cb_size_l_2) +
      " -DOCL_DIM_L_2=" + std::to_string(config.ocl_dim_l_2) +
      " -DNUM_WG_L_2=" + std::to_string(config.num_wg_l_2) +
      " -DNUM_WI_L_2=" + std::to_string(config.num_wi_l_2) +
      " -DINPUT_SIZE_R_1=" + std::to_string(config.input_size_r_1) +
      " -DL_CB_SIZE_R_1=" + std::to_string(config.l_cb_size_r_1) +
      " -DP_CB_SIZE_R_1=" + std::to_string(config.p_cb_size_r_1) +
      " -DOCL_DIM_R_1=" + std::to_string(config.ocl_dim_r_1) +
      " -DNUM_WG_R_1=" + std::to_string(config.num_wg_r_1) +
      " -DNUM_WI_R_1=" + std::to_string(config.num_wi_r_1) +
      " -DL_REDUCTION=" + std::to_string(config.l_reduction) +
      " -DP_WRITE_BACK=" + std::to_string(config.p_write_back) +
      " -DL_WRITE_BACK=" + std::to_string(config.l_write_back);

  err = clBuildProgram(program, 1, &device, compile_flags.c_str(), nullptr,
                       nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), nullptr);
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
  global_size[0] =
      ((config.ocl_dim_l_1 == 0) * config.num_wg_l_1 * config.num_wi_l_1 +
       (config.ocl_dim_l_2 == 0) * config.num_wg_l_2 * config.num_wi_l_2 +
       (config.ocl_dim_r_1 == 0) * config.num_wg_r_1 * config.num_wi_r_1);
  global_size[1] =
      ((config.ocl_dim_l_1 == 1) * config.num_wg_l_1 * config.num_wi_l_1 +
       (config.ocl_dim_l_2 == 1) * config.num_wg_l_2 * config.num_wi_l_2 +
       (config.ocl_dim_r_1 == 1) * config.num_wg_r_1 * config.num_wi_r_1);
  global_size[2] =
      ((config.ocl_dim_l_1 == 2) * config.num_wg_l_1 * config.num_wi_l_1 +
       (config.ocl_dim_l_2 == 2) * config.num_wg_l_2 * config.num_wi_l_2 +
       (config.ocl_dim_r_1 == 2) * config.num_wg_r_1 * config.num_wi_r_1);

  local_size[0] = ((config.ocl_dim_l_1 == 0) * config.num_wi_l_1 +
                   (config.ocl_dim_l_2 == 0) * config.num_wi_l_2 +
                   (config.ocl_dim_r_1 == 0) * config.num_wi_r_1);
  local_size[1] = ((config.ocl_dim_l_1 == 1) * config.num_wi_l_1 +
                   (config.ocl_dim_l_2 == 1) * config.num_wi_l_2 +
                   (config.ocl_dim_r_1 == 1) * config.num_wi_r_1);
  local_size[2] = ((config.ocl_dim_l_1 == 2) * config.num_wi_l_1 +
                   (config.ocl_dim_l_2 == 2) * config.num_wi_l_2 +
                   (config.ocl_dim_r_1 == 2) * config.num_wi_r_1);

  // Ensure local sizes are at least 1
  for (int i = 0; i < 3; ++i) {
    if (local_size[i] == 0) {
      // error case, exit
      std::cerr << "Error: Local size dimension " << i << " is zero."
                << std::endl;
      exit(1);
    }
  }

  std::cout << "Global size: " << global_size[0] << " x " << global_size[1]
            << " x " << global_size[2] << std::endl;
  std::cout << "Local size: " << local_size[0] << " x " << local_size[1]
            << " x " << local_size[2] << std::endl;
  record_phase("OpenCL setup + kernel preparation");

  // Execute gemm_1 kernel
  std::cout << "\n--- Executing gemm_1 kernel ---" << std::endl;
  cl_event event_1;
  err = clEnqueueNDRangeKernel(queue, kernel_1, 3, nullptr, global_size,
                               local_size, 0, nullptr, &event_1);
  if (err != CL_SUCCESS) {
    std::cerr << "Error enqueuing gemm_1 kernel: " << err << " ("
              << get_opencl_error_string(err) << ")" << std::endl;
    dump_system_diagnostics("gemm_1 enqueue failure");
    return EXIT_FAILURE;
  }

  err = clWaitForEvents(1, &event_1);
  if (err != CL_SUCCESS) {
    std::cerr << "Error waiting for gemm_1 kernel: " << err << " ("
              << get_opencl_error_string(err) << ")" << std::endl;
    dump_system_diagnostics("gemm_1 wait failure");
    return EXIT_FAILURE;
  }

  // Check if we need gemm_2 (reduction kernel) - needed when NUM_WG_R_1 > 1
  bool needs_reduction = (config.num_wg_r_1 >= 1);

  if (needs_reduction) {
    // Set kernel_2 arguments (int_res, res_g, c)
    clSetKernelArg(kernel_2, 0, sizeof(cl_mem), &buf_int_res);
    clSetKernelArg(kernel_2, 1, sizeof(cl_mem), &buf_res_g);
    clSetKernelArg(kernel_2, 2, sizeof(cl_mem), &buf_c);

    // Calculate global and local sizes for gemm_2 (reduction kernel)
    size_t global_size_2[3], local_size_2[3];
    global_size_2[0] =
        ((config.ocl_dim_l_1 == 0) * config.num_wg_l_1 * config.num_wi_l_1 +
         (config.ocl_dim_l_2 == 0) * config.num_wg_l_2 * config.num_wi_l_2 +
         (config.ocl_dim_r_1 == 0) * config.num_wi_r_1);
    global_size_2[1] =
        ((config.ocl_dim_l_1 == 1) * config.num_wg_l_1 * config.num_wi_l_1 +
         (config.ocl_dim_l_2 == 1) * config.num_wg_l_2 * config.num_wi_l_2 +
         (config.ocl_dim_r_1 == 1) * config.num_wi_r_1);
    global_size_2[2] =
        ((config.ocl_dim_l_1 == 2) * config.num_wg_l_1 * config.num_wi_l_1 +
         (config.ocl_dim_l_2 == 2) * config.num_wg_l_2 * config.num_wi_l_2 +
         (config.ocl_dim_r_1 == 2) * config.num_wi_r_1);

    local_size_2[0] = ((config.ocl_dim_l_1 == 0) * config.num_wi_l_1 +
                       (config.ocl_dim_l_2 == 0) * config.num_wi_l_2 +
                       (config.ocl_dim_r_1 == 0) * config.num_wi_r_1);
    local_size_2[1] = ((config.ocl_dim_l_1 == 1) * config.num_wi_l_1 +
                       (config.ocl_dim_l_2 == 1) * config.num_wi_l_2 +
                       (config.ocl_dim_r_1 == 1) * config.num_wi_r_1);
    local_size_2[2] = ((config.ocl_dim_l_1 == 2) * config.num_wi_l_1 +
                       (config.ocl_dim_l_2 == 2) * config.num_wi_l_2 +
                       (config.ocl_dim_r_1 == 2) * config.num_wi_r_1);

    // Ensure local sizes are at least 1
    for (int i = 0; i < 3; ++i) {
      if (local_size_2[i] == 0) {
        // error case, exit
        std::cerr << "Error: Local size dimension " << i
                  << " for gemm_2 is zero." << std::endl;
        exit(1);
      }
    }

    std::cout << "\n--- Executing gemm_2 (reduction) kernel ---" << std::endl;
    std::cout << "Global size: " << global_size_2[0] << " x "
              << global_size_2[1] << " x " << global_size_2[2] << std::endl;
    std::cout << "Local size: " << local_size_2[0] << " x " << local_size_2[1]
              << " x " << local_size_2[2] << std::endl;

    cl_event event_2;
    err = clEnqueueNDRangeKernel(queue, kernel_2, 3, nullptr, global_size_2,
                                 local_size_2, 0, nullptr, &event_2);
    if (err != CL_SUCCESS) {
      std::cerr << "Error enqueuing gemm_2 kernel: " << err << " ("
                << get_opencl_error_string(err) << ")" << std::endl;
      dump_system_diagnostics("gemm_2 enqueue failure");
      return EXIT_FAILURE;
    }

    err = clWaitForEvents(1, &event_2);
    if (err != CL_SUCCESS) {
      std::cerr << "Error waiting for gemm_2 kernel: " << err << " ("
                << get_opencl_error_string(err) << ")" << std::endl;
      dump_system_diagnostics("gemm_2 wait failure");
      return EXIT_FAILURE;
    }

    // Get kernel_2 execution time
    cl_ulong start_time_2, end_time_2;
    clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &start_time_2, nullptr);
    clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                            &end_time_2, nullptr);
    unsigned long long runtime_ns_2 = end_time_2 - start_time_2;

    std::cout << "Device gemm_2 kernel execution time: "
              << (runtime_ns_2 / 1000000.0) << " ms" << std::endl;
    clReleaseEvent(event_2);
  } else {
    std::cout << "No reduction needed (NUM_WG_R_1 = " << config.num_wg_r_1
              << ")" << std::endl;
  }

  // Read back Device results
  if (needs_reduction) {
    // If reduction was performed, result is in buf_c
    err = clEnqueueReadBuffer(queue, buf_c, CL_TRUE, 0, c.size() * sizeof(float),
                              c.data(), 0, nullptr, nullptr);
  } else {
    // If no reduction, result is in buf_res_g (should be same size as c when
    // NUM_WG_R_1 = 1)
    err = clEnqueueReadBuffer(queue, buf_res_g, CL_TRUE, 0,
                              c.size() * sizeof(float), c.data(), 0, nullptr,
                              nullptr);
  }

  if (err != CL_SUCCESS) {
    std::cerr << "Error reading GEMM output buffer: " << err << " ("
              << get_opencl_error_string(err) << ")" << std::endl;
    dump_system_diagnostics("gemm readback failure");
    return EXIT_FAILURE;
  }

  if (is_all_zero_output(c)) {
    std::cerr << "Error: GEMM output is all zeros. This may indicate a silent "
                 "runtime/device failure."
              << std::endl;
    dump_system_diagnostics("gemm all-zero output detected");
    return EXIT_FAILURE;
  }
  record_phase("Device execution + readback");

  // Compute CPU reference result
  std::vector<float> ref_c(M * N);
  std::cout << "Computing reference result on CPU..." << std::endl;

  cl_ulong reference_start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  compute_reference(a, b, ref_c, M, N, K);
  cl_ulong reference_end_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  unsigned long long reference_runtime_ns = reference_end_time - reference_start_time;

  // Get gemm_1 kernel execution time
  cl_ulong start_time, end_time;
  clGetEventProfilingInfo(event_1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &start_time, nullptr);
  clGetEventProfilingInfo(event_1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &end_time, nullptr);
  unsigned long long runtime_ns = end_time - start_time;
  record_phase("CPU reference compute");

  std::cout << "CPU reference execution time: " << (reference_runtime_ns / 1000000.0)
            << " ms" << std::endl;
  std::cout << "Device kernel execution time: " << (runtime_ns / 1000000.0)
            << " ms" << std::endl;
  std::cout << "Speedup (gemm_1 vs CPU reference): "
            << static_cast<double>(reference_runtime_ns) / runtime_ns << "x"
            << std::endl;

  // Validate results
  std::cout << "\n" << std::string(50, '=') << std::endl;
  auto validation_start_time = std::chrono::high_resolution_clock::now();
  bool valid = validate_result(c, ref_c, M * N);
  auto validation_end_time = std::chrono::high_resolution_clock::now();
  auto validation_runtime_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          validation_end_time - validation_start_time)
          .count();
  std::cout << "Reference check execution time: "
            << (validation_runtime_ns / 1000000.0) << " ms" << std::endl;
  std::cout << std::string(50, '=') << std::endl;
  record_phase("Reference check");

  if (!valid) {
    dump_system_diagnostics("gemm result validation mismatch");
  }

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
  record_phase("Cleanup");

  double accounted_ms = 0.0;
  for (const auto &phase_timing : phase_timings_ms) {
    accounted_ms += phase_timing.second;
  }
  const auto total_program_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          WallClock::now() - program_start_time)
          .count();

  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << "Coarse timing breakdown (wall-clock):" << std::endl;
  for (const auto &phase_timing : phase_timings_ms) {
    std::cout << "  " << phase_timing.first << ": " << phase_timing.second
              << " ms" << std::endl;
  }
  std::cout << "  Accounted total: " << accounted_ms << " ms" << std::endl;
  std::cout << "  Whole program:   " << total_program_ms << " ms"
            << std::endl;
  std::cout << "  Unaccounted:     " << (total_program_ms - accounted_ms)
            << " ms" << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
