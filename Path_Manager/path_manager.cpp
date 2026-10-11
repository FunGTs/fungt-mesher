#include "path_manager.hpp"

#include <algorithm>

std::filesystem::path findProjectRoot() {
  std::filesystem::path root(FGT_MESHER_BASE_DIR);

  // Fail immediately if the path provided to CMake doesn't hold the config
  if (!std::filesystem::exists(root / "fungt-mesher.config")) {
    throw std::runtime_error(
        "Path Error: 'fungt-mesher.config' not found at FGT_MESHER_BASE_DIR: " +
        root.string());
  }

  return std::filesystem::canonical(root);
}

std::string getAssetPath(const std::string &path) {
  std::filesystem::path root = findProjectRoot();

  // Safely merges paths, fixing slash bugs automatically
  std::filesystem::path full_path =
      std::filesystem::weakly_canonical(root / path);

  // Guard: Confirms that resolving the asset path didn't try to escape the base
  // directory
  auto [root_it, full_it] = std::mismatch(root.begin(), root.end(),
                                          full_path.begin(), full_path.end());
  if (root_it != root.end()) {
    throw std::runtime_error("Security Exception: Attempted path traversal "
                             "outside of FGT_MESHER_BASE_DIR!");
  }

  return full_path.string();
}
