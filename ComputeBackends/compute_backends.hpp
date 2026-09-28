#if !defined(_COMPUTE_BACKENDS)
#define _COMPUTE_BACKENDS
#include <iostream>
#include <string>

enum class Backend {
  SYCL_CUDA,
  SYCL_LEVEL_ZERO,
  SYCL_OPENCL, // SYCL targeting CUDA devices
};

class ComputeBackend {
public:
  static const std::string GetBackendName();
  static void SetBackend(Backend api) { s_API = api; }
  static Backend GetBackend() { return s_API; }

private:
  static Backend s_API;
};

#endif // _COMPUTE_BACKENDS
