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

constexpr std::size_t element_count = 1024 * 3072;
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0]
              << " <funlib_output.bin> [reference_output.bin]\n";
    return 2;
  }

  try {
    const std::filesystem::path actual_path = argv[1];
    const std::filesystem::path reference_path =
        argc == 3 ? argv[2] : "decoder_refs/final_output.bin";
    const auto actual = load_tensor(actual_path);
    const auto reference = load_tensor(reference_path);

    double maximum = 0.0;
    double difference_sum = 0.0;
    std::size_t worst_index = 0;
    std::size_t outside_tolerance = 0;

    for (std::size_t index = 0; index < element_count; ++index) {
      if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
        maximum = std::numeric_limits<double>::infinity();
        worst_index = index;
        ++outside_tolerance;
        continue;
      }

      const double difference =
          std::abs(static_cast<double>(actual[index]) - reference[index]);
      difference_sum += difference;
      if (difference > maximum) {
        maximum = difference;
        worst_index = index;
      }
      if (difference > threshold) {
        ++outside_tolerance;
      }
    }

    std::cout << std::setprecision(10);
    std::cout << "Shape: [1, 1024, 3072]\n";
    std::cout << "Tolerance: " << threshold << '\n';
    std::cout << "Max absolute difference: " << maximum << '\n';
    std::cout << "Mean absolute difference: "
              << difference_sum / element_count << '\n';
    std::cout << "Worst position: channel " << worst_index / 3072
              << ", token " << worst_index % 3072 << '\n';
    std::cout << "Actual: " << actual[worst_index] << '\n';
    std::cout << "Reference: " << reference[worst_index] << '\n';
    std::cout << "Values outside tolerance: " << outside_tolerance << " / "
              << element_count << '\n';

    if (outside_tolerance != 0) {
      std::cout << "FAIL\n";
      return 1;
    }

    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
