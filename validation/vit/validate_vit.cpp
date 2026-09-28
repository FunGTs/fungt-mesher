#include <cmath>
#include <cstddef>
#include <cstdlib>
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
constexpr double default_tolerance = 1.0e-3;

std::vector<float> load_tensor(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open " + path);
  }

  const auto expected_bytes =
      static_cast<std::streamsize>(element_count * sizeof(float));
  if (file.tellg() != std::streampos(expected_bytes)) {
    throw std::runtime_error(path + " must contain exactly " +
                             std::to_string(expected_bytes) + " bytes");
  }

  file.seekg(0);
  std::vector<float> values(element_count);
  if (!file.read(reinterpret_cast<char *>(values.data()), expected_bytes)) {
    throw std::runtime_error("Cannot read " + path);
  }
  return values;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " [actual.bin] [reference.bin] [tolerance]\n";
    return 2;
  }

  try {
    const std::string actual_path =
        argc >= 2 ? argv[1] : "image_features.bin";
    const std::string reference_path =
        argc >= 3 ? argv[2] : "ref_vit_output.bin";
    const double tolerance =
        argc >= 4 ? std::stod(argv[3]) : default_tolerance;
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
      throw std::invalid_argument("Tolerance must be finite and non-negative");
    }

    const auto actual = load_tensor(actual_path);
    const auto reference = load_tensor(reference_path);

    double max_difference = 0.0;
    double difference_sum = 0.0;
    std::size_t worst_index = 0;
    std::size_t mismatches = 0;

    for (std::size_t index = 0; index < element_count; ++index) {
      if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
        throw std::runtime_error("Non-finite value at index " +
                                 std::to_string(index));
      }

      const double difference = std::abs(
          static_cast<double>(actual[index]) - reference[index]);
      difference_sum += difference;
      if (difference > max_difference) {
        max_difference = difference;
        worst_index = index;
      }
      if (difference > tolerance) {
        ++mismatches;
      }
    }

    const std::size_t worst_token = worst_index / embedding_size;
    const std::size_t worst_feature = worst_index % embedding_size;
    const double mean_difference = difference_sum / element_count;
    const bool passed = mismatches == 0;

    std::cout << std::setprecision(10)
              << "Shape: [" << token_count << ", " << embedding_size << "]\n"
              << "Tolerance: " << tolerance << '\n'
              << "Max absolute difference: " << max_difference << '\n'
              << "Mean absolute difference: " << mean_difference << '\n'
              << "Worst position: token " << worst_token << ", feature "
              << worst_feature << '\n'
              << "Actual: " << actual[worst_index] << '\n'
              << "Reference: " << reference[worst_index] << '\n'
              << "Values outside tolerance: " << mismatches << " / "
              << element_count << '\n'
              << (passed ? "PASS" : "FAIL") << '\n';

    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
