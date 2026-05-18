#include <CL/cl_platform.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

using json = nlohmann::json;

// Helper function to decode OpenCL error codes
const char *get_opencl_error_string(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  default:
    return "UNKNOWN_ERROR (vendor-specific or invalid error code)";
  }
}

// Tuning parameters and input size struct
struct GaussianConfig {
  // Input dimensions
  int input_size_h = -1;
  int input_size_w = -1;

  // Tuning parameters with default values
  int g_cb_res_dest_level = -1;
  int l_cb_res_dest_level = -1;
  int p_cb_res_dest_level = -1;
  int images_cache_lcl = -1;
  int images_cache_prv = -1;
  int filter_cache_lcl = -1;
  int filter_cache_prv = -1;
  int out_cache_prv = -1;
  int wg_1_ocl_dim = -1;
  int wg_2_ocl_dim = -1;
  int wi_1_ocl_dim = -1;
  int wi_2_ocl_dim = -1;
  int glb_1 = -1;
  int wg_1 = -1;
  int lcl_1 = -1;
  int wi_1 = -1;
  int prv_1 = -1;
  int glb_2 = -1;
  int wg_2 = -1;
  int lcl_2 = -1;
  int wi_2 = -1;
  int prv_2 = -1;
};

// Load configuration from JSON file
GaussianConfig load_config_from_json(const std::string &json_path) {
  GaussianConfig config;

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
    if (j.contains("input_size_h"))
      config.input_size_h = j["input_size_h"];
    if (j.contains("input_size_w"))
      config.input_size_w = j["input_size_w"];

    // Load tuning parameters
    if (j.contains("g_cb_res_dest_level"))
      config.g_cb_res_dest_level = j["g_cb_res_dest_level"];
    if (j.contains("l_cb_res_dest_level"))
      config.l_cb_res_dest_level = j["l_cb_res_dest_level"];
    if (j.contains("p_cb_res_dest_level"))
      config.p_cb_res_dest_level = j["p_cb_res_dest_level"];
    if (j.contains("images_cache_lcl"))
      config.images_cache_lcl = j["images_cache_lcl"];
    if (j.contains("images_cache_prv"))
      config.images_cache_prv = j["images_cache_prv"];
    if (j.contains("filter_cache_lcl"))
      config.filter_cache_lcl = j["filter_cache_lcl"];
    if (j.contains("filter_cache_prv"))
      config.filter_cache_prv = j["filter_cache_prv"];
    if (j.contains("out_cache_prv"))
      config.out_cache_prv = j["out_cache_prv"];
    if (j.contains("wg_1_ocl_dim"))
      config.wg_1_ocl_dim = j["wg_1_ocl_dim"];
    if (j.contains("wg_2_ocl_dim"))
      config.wg_2_ocl_dim = j["wg_2_ocl_dim"];
    if (j.contains("wi_1_ocl_dim"))
      config.wi_1_ocl_dim = j["wi_1_ocl_dim"];
    if (j.contains("wi_2_ocl_dim"))
      config.wi_2_ocl_dim = j["wi_2_ocl_dim"];
    if (j.contains("glb_1"))
      config.glb_1 = j["glb_1"];
    if (j.contains("wg_1"))
      config.wg_1 = j["wg_1"];
    if (j.contains("lcl_1"))
      config.lcl_1 = j["lcl_1"];
    if (j.contains("wi_1"))
      config.wi_1 = j["wi_1"];
    if (j.contains("prv_1"))
      config.prv_1 = j["prv_1"];
    if (j.contains("glb_2"))
      config.glb_2 = j["glb_2"];
    if (j.contains("wg_2"))
      config.wg_2 = j["wg_2"];
    if (j.contains("lcl_2"))
      config.lcl_2 = j["lcl_2"];
    if (j.contains("wi_2"))
      config.wi_2 = j["wi_2"];
    if (j.contains("prv_2"))
      config.prv_2 = j["prv_2"];

  } catch (const json::exception &e) {
    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    std::cerr << "Using default parameters." << std::endl;
    exit(EXIT_FAILURE);
  }

  return config;
}

std::string
get_kernel_template_path(const char *kernel_name = "gaussian_static_1.cl") {
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

bool dump_opencl_program_binary(cl_program program, cl_device_id device,
                                const std::string &output_path) {
  size_t binary_size = 0;
  cl_int err =
      clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size),
                       &binary_size, nullptr);
  if (err != CL_SUCCESS || binary_size == 0) {
    std::cerr << "Error querying OpenCL binary size: " << err << std::endl;
    return false;
  }

  std::vector<unsigned char> binary(binary_size);
  unsigned char *binary_ptr = binary.data();
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binary_ptr),
                         &binary_ptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Error querying OpenCL binary: " << err << std::endl;
    return false;
  }

  namespace fs = std::filesystem;
  fs::path path(output_path);
  if (path.has_parent_path()) {
    fs::create_directories(path.parent_path());
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Error: Could not open OpenCL binary output: " << output_path
              << std::endl;
    return false;
  }
  out.write(reinterpret_cast<const char *>(binary.data()), binary.size());
  if (!out.good()) {
    std::cerr << "Error writing OpenCL binary output: " << output_path
              << std::endl;
    return false;
  }
  return true;
}

// CPU reference: 5x5 filter from gaussian_static_1.cl kernel
void compute_reference(const std::vector<float> &in_padded,
                       std::vector<float> &out, size_t H, size_t W) {
  // 5x5 filter kernel from the OpenCL code
  // Format: kernel[dy][dx] where (0,0) is top-left corner
  float kernel[5][5] = {{2.0f, 4.0f, 5.0f, 4.0f, 2.0f},
                        {4.0f, 9.0f, 12.0f, 9.0f, 4.0f},
                        {5.0f, 12.0f, 15.0f, 12.0f, 5.0f},
                        {4.0f, 9.0f, 12.0f, 9.0f, 4.0f},
                        {2.0f, 4.0f, 5.0f, 4.0f, 2.0f}};

  for (size_t y = 0; y < H; ++y) {
    for (size_t x = 0; x < W; ++x) {
      float result = 0.0f;

      // Apply 5x5 convolution
      for (int dy = 0; dy < 5; ++dy) {
        for (int dx = 0; dx < 5; ++dx) {
          size_t iy = y + dy; // Input is padded, so y+dy gives correct position
          size_t ix = x + dx;
          result += in_padded[iy * (W + 4) + ix] * kernel[dy][dx];
        }
      }

      out[y * W + x] = result;
    }
  }
}

// Compare Device result with reference (with floating-point tolerance)
bool validate_result(const std::vector<float> &device_out,
                     const std::vector<float> &ref_out, size_t size) {
  int mismatch_count = 0;
  float max_diff = 0.0f;
  const float tolerance = 1e-2f; // Allow small floating-point errors

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
  std::string config_path;
  std::string dump_opencl_binary_path;
  bool dump_opencl_binary_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dump-opencl-binary") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --dump-opencl-binary requires a path."
                  << std::endl;
        return EXIT_FAILURE;
      }
      dump_opencl_binary_path = argv[++i];
    } else if (arg == "--dump-opencl-binary-only") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --dump-opencl-binary-only requires a path."
                  << std::endl;
        return EXIT_FAILURE;
      }
      dump_opencl_binary_path = argv[++i];
      dump_opencl_binary_only = true;
    } else if (config_path.empty()) {
      config_path = arg;
    } else {
      std::cerr << "Error: Unexpected argument: " << arg << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (config_path.empty()) {
    std::cout << "No config file provided. Please provide a configuration JSON "
                 "file as an argument."
              << std::endl;
    return EXIT_FAILURE;
  }

  GaussianConfig config = load_config_from_json(config_path);
  record_phase("Arg parsing + config load");

  std::cout << "Using configuration from: " << config_path << std::endl;
  std::cout << "Input size: " << config.input_size_h << " x "
            << config.input_size_w << std::endl;
  std::cout << "Tuning parameters:" << std::endl;
  std::cout << "  G_CB_RES_DEST_LEVEL: " << config.g_cb_res_dest_level
            << std::endl;
  std::cout << "  L_CB_RES_DEST_LEVEL: " << config.l_cb_res_dest_level
            << std::endl;
  std::cout << "  WG_1: " << config.wg_1 << ", WI_1: " << config.wi_1
            << std::endl;
  std::cout << "  WG_2: " << config.wg_2 << ", WI_2: " << config.wi_2
            << std::endl;

  const size_t H = config.input_size_h;
  const size_t W = config.input_size_w;
  cl_int err;

  // Get platform and CPU device
  cl_platform_id platform;
  cl_device_id device;
  if (!select_cpu_platform_and_device(platform, device)) {
    return EXIT_FAILURE;
  }

  char platform_name[256];
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
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

  // Create deterministic integer-valued input data (stored as float) to keep
  // verification stable and reproducible.

  std::vector<float> in((H + 4) * (W + 4));
  for (size_t i = 0; i < in.size(); ++i)
    in[i] = static_cast<float>((i % 11) + 1); // [1, 11]

  std::vector<float> out(H * W, 0);
  std::vector<float> dummy(H * W, 0); // Dummy buffer (unused in kernel)

  cl_mem buf_in =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     in.size() * sizeof(float), in.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating input buffer: " << err << std::endl;
    return EXIT_FAILURE;
  }

  cl_mem buf_dummy = clCreateBuffer(
      context, CL_MEM_READ_WRITE, dummy.size() * sizeof(float), nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating dummy buffer: " << err << std::endl;
    return EXIT_FAILURE;
  }

  cl_mem buf_out =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     out.size() * sizeof(float), out.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating output buffer: " << err << std::endl;
    return EXIT_FAILURE;
  }

  // Load kernel from template using relative path
  std::string kernel_template_path =
      get_kernel_template_path("gaussian_static_1.cl");
  std::string kernel_source = read_kernel_source(kernel_template_path.c_str());
  const char *src = kernel_source.c_str();
  cl_program program =
      clCreateProgramWithSource(context, 1, &src, nullptr, &err);

  std::string compile_flags =
      "-DTYPE_T=float -DTYPE_TS=float "
      "-DG_CB_RES_DEST_LEVEL=" +
      std::to_string(config.g_cb_res_dest_level) +
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

  if (!dump_opencl_binary_path.empty() &&
      !dump_opencl_program_binary(program, device, dump_opencl_binary_path)) {
    return EXIT_FAILURE;
  }
  if (dump_opencl_binary_only) {
    std::cout << "Dump-only mode completed; kernel execution skipped."
              << std::endl;
    clReleaseProgram(program);
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_dummy);
    clReleaseMemObject(buf_out);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return EXIT_SUCCESS;
  }

  // Create kernel
  cl_kernel kernel = clCreateKernel(program, "gaussian_1", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Error creating kernel: " << err << std::endl;
    return EXIT_FAILURE;
  }

  // Set kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
  if (err != CL_SUCCESS) {
    std::cerr << "Error setting kernel argument 0: " << err << std::endl;
    return EXIT_FAILURE;
  }

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dummy);
  if (err != CL_SUCCESS) {
    std::cerr << "Error setting kernel argument 1: " << err << std::endl;
    return EXIT_FAILURE;
  }

  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
  if (err != CL_SUCCESS) {
    std::cerr << "Error setting kernel argument 2: " << err << std::endl;
    return EXIT_FAILURE;
  }

  // Calculate global and local sizes
  size_t global_size[2], local_size[2];
  global_size[0] = ((config.wg_1_ocl_dim == 0) * config.wg_1 +
                    (config.wg_2_ocl_dim == 0) * config.wg_2) *
                   ((config.wi_1_ocl_dim == 0) * config.wi_1 +
                    (config.wi_2_ocl_dim == 0) * config.wi_2);
  global_size[1] = ((config.wg_1_ocl_dim == 1) * config.wg_1 +
                    (config.wg_2_ocl_dim == 1) * config.wg_2) *
                   ((config.wi_1_ocl_dim == 1) * config.wi_1 +
                    (config.wi_2_ocl_dim == 1) * config.wi_2);
  local_size[0] = (config.wi_1_ocl_dim == 0) * config.wi_1 +
                  (config.wi_2_ocl_dim == 0) * config.wi_2;
  local_size[1] = (config.wi_1_ocl_dim == 1) * config.wi_1 +
                  (config.wi_2_ocl_dim == 1) * config.wi_2;

  // Ensure local sizes are at least 1
  if (local_size[0] == 0 || local_size[1] == 0) {
    std::cerr
        << "Error: Local size dimensions cannot be zero. Wrong configuration"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Global size: " << global_size[0] << " x " << global_size[1]
            << std::endl;
  std::cout << "Local size: " << local_size[0] << " x " << local_size[1]
            << std::endl;

  // Query device limits to validate configuration
  size_t max_work_group_size;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                  &max_work_group_size, nullptr);
  size_t requested_work_group_size = local_size[0] * local_size[1];

  if (requested_work_group_size > max_work_group_size) {
    std::cerr << "Error: Requested work group size ("
              << requested_work_group_size << ") exceeds device maximum ("
              << max_work_group_size << ")" << std::endl;
    return EXIT_FAILURE;
  }

  cl_ulong local_mem_size, required_local_mem;
  clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
                  &local_mem_size, nullptr);
  std::cout << "Device local memory: " << (local_mem_size / 1024) << " KB"
            << std::endl;
  std::cout << "Max work group size: " << max_work_group_size << std::endl;
  record_phase("OpenCL setup + kernel preparation");

  // Execute kernel
  cl_event event;
  err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size,
                               local_size, 0, nullptr, &event);
  if (err != CL_SUCCESS) {
    std::cerr << "Error enqueuing kernel: " << err << " ("
              << get_opencl_error_string(err) << ")" << std::endl;
    if (err == CL_INVALID_WORK_GROUP_SIZE) {
      std::cerr << "  Reason: Invalid work group size" << std::endl;
    } else if (err == CL_OUT_OF_RESOURCES) {
      std::cerr << "  Reason: Out of resources (possibly too much local memory)"
                << std::endl;
    } else if (err == CL_OUT_OF_HOST_MEMORY) {
      std::cerr << "  Reason: Out of host memory" << std::endl;
    }
    dump_system_diagnostics("clEnqueueNDRangeKernel failure");
    return EXIT_FAILURE;
  }

  // Wait for kernel execution to complete and check for errors
  err = clWaitForEvents(1, &event);
  if (err != CL_SUCCESS) {
    std::cerr << "Error waiting for kernel execution: " << err << " ("
              << get_opencl_error_string(err) << ")" << std::endl;
    std::cerr << "  This often indicates a kernel crash or runtime failure."
              << std::endl;
    std::cerr << "  Possible causes:" << std::endl;
    std::cerr << "    - Illegal memory access (buffer overflow)" << std::endl;
    std::cerr << "    - Division by zero in kernel" << std::endl;
    std::cerr << "    - Stack overflow (too much private memory: "
              << config.prv_1 << " x " << config.prv_2 << " = "
              << (config.prv_1 * config.prv_2) << " elements per work item)"
              << std::endl;
    std::cerr << "    - Invalid work group configuration" << std::endl;
    dump_system_diagnostics("clWaitForEvents failure");
    return EXIT_FAILURE;
  }

  // Check kernel execution status
  cl_int exec_status;
  err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                       &exec_status, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Error getting kernel execution status: " << err << std::endl;
    return EXIT_FAILURE;
  }

  if (exec_status < 0) {
    std::cerr << "Kernel execution failed with status: " << exec_status
              << std::endl;
    std::cerr << "  Negative status indicates execution error (kernel crashed)"
              << std::endl;
    dump_system_diagnostics("negative kernel execution status");
    return EXIT_FAILURE;
  }

  if (exec_status != CL_COMPLETE) {
    std::cerr << "Kernel execution incomplete with status: " << exec_status
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "✓ Kernel executed successfully" << std::endl;

  // Read back Device result
  err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0,
                            out.size() * sizeof(float), out.data(), 0, nullptr,
                            nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Error reading output buffer: " << err << std::endl;
    dump_system_diagnostics("clEnqueueReadBuffer failure");
    return EXIT_FAILURE;
  }

  if (is_all_zero_output(out)) {
    std::cerr << "Error: Device output is all zeros. This may indicate a "
                 "silent runtime/device failure."
              << std::endl;
    dump_system_diagnostics("all-zero output detected");
    return EXIT_FAILURE;
  }
  record_phase("Device execution + readback");

  // Compute CPU reference result
  std::vector<float> ref_out(H * W);
  std::cout << "Computing reference result on CPU..." << std::endl;

  cl_ulong ref_start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  compute_reference(in, ref_out, H, W);
  cl_ulong ref_end_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  unsigned long long ref_runtime_ns = ref_end_time - ref_start_time;

  // Get kernel execution time
  cl_ulong start_time, end_time;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &start_time, nullptr);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &end_time, nullptr);
  unsigned long long runtime_ns = end_time - start_time;
  record_phase("CPU reference compute");

  std::cout << "Reference CPU execution time: " << (ref_runtime_ns / 1000000.0)
            << " ms" << std::endl;
  std::cout << "Device kernel execution time: " << (runtime_ns / 1000000.0)
            << " ms" << std::endl;
  std::cout << "Speedup: " << (static_cast<double>(ref_runtime_ns) / runtime_ns)
            << "x" << std::endl;

  // Validate results
  std::cout << "\n" << std::string(50, '=') << std::endl;
  auto validation_start_time = std::chrono::high_resolution_clock::now();
  bool valid = validate_result(out, ref_out, H * W);
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
    dump_system_diagnostics("result validation mismatch");
  }

  // Cleanup
  clReleaseEvent(event);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(buf_in);
  clReleaseMemObject(buf_dummy);
  clReleaseMemObject(buf_out);
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
