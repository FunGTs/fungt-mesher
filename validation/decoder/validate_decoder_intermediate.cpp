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

constexpr std::size_t channel_count = 1024;
constexpr std::size_t token_count = 3072;
constexpr std::size_t element_count = channel_count * token_count;
constexpr double threshold = 1.0e-2;

std::vector<float> load_tensor(const std::filesystem::path &path) {
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

double max_absolute_difference(const std::vector<float> &reference,
                               const std::vector<float> &actual) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < element_count; ++index) {
    if (!std::isfinite(reference[index]) || !std::isfinite(actual[index])) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(
        maximum,
        std::abs(static_cast<double>(actual[index]) - reference[index]));
  }
  return maximum;
}

bool validate_stage(const std::filesystem::path &reference_directory,
                    const std::filesystem::path &actual_directory,
                    const std::string &stage) {
  const auto reference =
      load_tensor(reference_directory / (stage + ".bin"));
  const auto actual =
      load_tensor(actual_directory / ("funlib_" + stage + ".bin"));
  const double difference = max_absolute_difference(reference, actual);

  std::cout << std::left << std::setw(18) << stage
            << " max absolute difference: " << std::setprecision(10)
            << difference << '\n';

  if (difference > threshold) {
    std::cout << "FAIL: first divergence at " << stage
              << " (threshold " << threshold << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " [funlib_output_directory] [reference_directory]\n";
    return 2;
  }

  try {
    const std::filesystem::path actual_directory = argc >= 2 ? argv[1] : ".";
    const std::filesystem::path reference_directory =
        argc == 3 ? argv[2] : "decoder_refs";

    if (!validate_stage(reference_directory, actual_directory, "after_norm")) {
      return 1;
    }
    if (!validate_stage(reference_directory, actual_directory,
                        "after_proj_in")) {
      return 1;
    }

    for (std::size_t block = 0; block < 16; ++block) {
      if (!validate_stage(reference_directory, actual_directory,
                          "after_block" + std::to_string(block))) {
        return 1;
      }
    }

    if (!validate_stage(reference_directory, actual_directory,
                        "after_proj_out")) {
      return 1;
    }
    if (!validate_stage(reference_directory, actual_directory,
                        "final_output")) {
      return 1;
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
