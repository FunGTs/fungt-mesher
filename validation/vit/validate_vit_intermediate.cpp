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

constexpr std::size_t token_count = 197;
constexpr std::size_t embedding_size = 768;
constexpr std::size_t element_count = token_count * embedding_size;
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

double compare(const std::vector<float> &reference,
               const std::vector<float> &actual) {
  double max_difference = 0.0;
  for (std::size_t index = 0; index < element_count; ++index) {
    if (!std::isfinite(reference[index]) || !std::isfinite(actual[index])) {
      return std::numeric_limits<double>::infinity();
    }
    max_difference = std::max(
        max_difference,
        std::abs(static_cast<double>(actual[index]) - reference[index]));
  }
  return max_difference;
}

bool validate_stage(const std::filesystem::path &directory,
                    const std::string &stage) {
  const auto reference =
      load_tensor(directory / ("ref_after_" + stage + ".bin"));
  const auto actual =
      load_tensor(directory / ("funlib_after_" + stage + ".bin"));
  const double max_difference = compare(reference, actual);

  std::cout << std::left << std::setw(14) << stage
            << " max absolute difference: " << std::setprecision(10)
            << max_difference << '\n';
  if (max_difference > threshold) {
    std::cout << "FAIL: first divergence at " << stage
              << " (threshold " << threshold << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [validation_directory]\n";
    return 2;
  }

  try {
    const std::filesystem::path directory = argc == 2 ? argv[1] : ".";

    if (!validate_stage(directory, "embeddings")) {
      return 1;
    }
    for (std::size_t layer = 0; layer < 12; ++layer) {
      if (!validate_stage(directory, "layer" + std::to_string(layer))) {
        return 1;
      }
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
