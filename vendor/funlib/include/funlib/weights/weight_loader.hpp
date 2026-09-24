#ifndef FUNLIB_WEIGHT_LOADER_HPP
#define FUNLIB_WEIGHT_LOADER_HPP

#include <filesystem>
#include <funlib/Tensor/tensor.hpp>
#include <optional>
#include <string>
#include <unordered_map>

namespace flib {

class WeightLoader {
public:
  using WeightMap = std::unordered_map<std::string, Tensor<float>>;

  // Without a queue, tensors reside on the host. With a queue, uploads finish
  // before construction returns. Binary files are little-endian float32.
  explicit WeightLoader(const std::filesystem::path &manifest,
                        std::optional<sycl::queue> queue = std::nullopt);

  const Tensor<float> &at(const std::string &name) const;
  bool contains(const std::string &name) const;
  std::size_t size() const noexcept { return weights_.size(); }
  const WeightMap &weights() const noexcept { return weights_; }

private:
  WeightMap weights_;
};

} // namespace flib
#endif
