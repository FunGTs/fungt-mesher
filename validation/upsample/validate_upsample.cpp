#include <funlib/funlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t plane_count = 3;
constexpr std::size_t input_channels = 1024;
constexpr std::size_t input_size = 32;
constexpr std::size_t output_channels = 40;
constexpr std::size_t output_size = 64;
constexpr std::size_t detokenize_count =
    plane_count * input_channels * input_size * input_size;
constexpr std::size_t upsample_count =
    plane_count * output_channels * output_size * output_size;
constexpr double threshold = 1.0e-2;

std::vector<float> load_tensor(const std::filesystem::path &path,
                               std::size_t element_count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open " + path.string());
  }

  const auto expected_bytes =
      static_cast<std::streamsize>(element_count * sizeof(float));
  if (file.tellg() != std::streampos(expected_bytes)) {
    throw std::runtime_error(path.string() + " must contain exactly " +
                             std::to_string(expected_bytes) + " bytes");
  }

  file.seekg(0);
  std::vector<float> values(element_count);
  if (!file.read(reinterpret_cast<char *>(values.data()), expected_bytes)) {
    throw std::runtime_error("Cannot read " + path.string());
  }
  return values;
}

flib::ftensor upload(const std::vector<std::size_t> &shape,
                     const std::vector<float> &values, sycl::queue &queue) {
  flib::ftensor tensor(shape, queue);
  tensor.copy_from(values.data(), queue).wait_and_throw();
  return tensor;
}

bool compare_stage(const std::string &stage,
                   const std::vector<float> &actual,
                   const std::vector<float> &reference) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
      maximum = std::numeric_limits<double>::infinity();
      break;
    }
    maximum = std::max(
        maximum,
        std::abs(static_cast<double>(actual[index]) - reference[index]));
  }

  std::cout << std::left << std::setw(12) << stage
            << " max absolute difference: " << std::setprecision(10)
            << maximum << '\n';

  if (maximum > threshold) {
    std::cout << stage << " FAIL\n";
    return false;
  }

  std::cout << stage << " PASS\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model_directory> <decoder_output.bin> "
                 "[upsample_reference_directory]\n";
    return 2;
  }

  try {
    const std::filesystem::path model_directory = argv[1];
    const std::filesystem::path decoder_output = argv[2];
    const std::filesystem::path reference_directory =
        argc == 4 ? argv[3] : "upsample_refs";

    flib::sycl_handler::register_queue("cuda", flib::device::GPU,
                                       flib::vendor::NVIDIA,
                                       flib::backend::CUDA, true);
    sycl::queue queue = flib::sycl_handler::get_queue("cuda");

    // Decoder output: [B, Ct, Np*Hp*Wp] = [1, 1024, 3072].
    auto decoder = upload(
        {1, input_channels, plane_count * input_size * input_size},
        load_tensor(decoder_output, detokenize_count), queue);

    // [1,1024,3072] -> [1,1024,3,32,32] -> [1,3,1024,32,32].
    decoder.reshape(
        {1, input_channels, plane_count, input_size, input_size});
    auto detokenized = flib::tensor_operations::permute(
        decoder, {0, 2, 1, 3, 4}, queue);

    const auto detokenize_reference = load_tensor(
        reference_directory / "ref_detokenize_output.bin", detokenize_count);
    if (!compare_stage("detokenize", detokenized.to_host(queue),
                       detokenize_reference)) {
      return 1;
    }

    // Merge B and Np so each plane is one NCHW batch element.
    detokenized.reshape(
        {plane_count, input_channels, input_size, input_size});

    // Load the real IOHW transposed-convolution weight and Cout bias.
    flib::WeightLoader weights(model_directory / "weights.json", queue);
    const auto &weight = weights.at("post_processor.upsample.weight");
    const auto &bias = weights.at("post_processor.upsample.bias");

    // Kernel=2, stride=2, padding=0, output_padding=0, dilation=1.
    auto upsampled = flib::operations::convolution2dTranspose(
        detokenized, weight, bias, 2, 0, 0, 1, queue);
    upsampled.reshape(
        {1, plane_count, output_channels, output_size, output_size});

    const auto upsample_reference = load_tensor(
        reference_directory / "ref_upsample_output.bin", upsample_count);
    if (!compare_stage("upsample", upsampled.to_host(queue),
                       upsample_reference)) {
      return 1;
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
