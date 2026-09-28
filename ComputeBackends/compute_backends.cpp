#include "compute_backends.hpp"

Backend ComputeBackend::s_API = Backend::SYCL_OPENCL; // default backend

const std::string ComputeBackend::GetBackendName() {
  switch (s_API) {
  case Backend::SYCL_CUDA:
    return "SYCL_CUDA";
  case Backend::SYCL_LEVEL_ZERO:
    return "SYCL_LEVEL_ZERO";
  case Backend::SYCL_OPENCL:
    return "SYCL_OPENCL";
  default:
    return "Unknown";
  }
}
