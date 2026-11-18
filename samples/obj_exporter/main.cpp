#include "FunGTMesher/fungt_mesher.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 4 || std::string(argv[2]) != "--out") {
    std::cerr << "Usage: " << argv[0]
              << " <image_path> --out <output_path>" << std::endl;
    return 1;
  }

  ComputeBackend::SetBackend(Backend::SYCL_CUDA);
  auto mesher = FunGTMesher::create_mesher();
  mesher->set_model();

  mesher->generate(argv[1], argv[3],
                   ImageResolution::HighResolution512);

  return 0;
}
