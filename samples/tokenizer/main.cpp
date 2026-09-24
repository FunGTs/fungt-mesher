#include <iostream>
#include "FunGTMesher/fungt_mesher.hpp"

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <image_path> <output_path>" << std::endl;
        return 1;
    }
    ComputeBackend::SetBackend(Backend::SYCL_CUDA);
    auto mesher = FunGTMesher::create_mesher();
    mesher->set_model(argv[1]);
    mesher->generate(argv[2], argv[3]);

    return 0;
}
