#if !defined(_WEIGHT_LOADER_HPP_)
#define _WEIGHT_LOADER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fgtm_tools {

class WeightLoader;

class ScopedWeightLoader {
private:
  WeightLoader &m_loader;
  std::string m_prefix;

public:
  ScopedWeightLoader(WeightLoader &loader, const std::string &prefix);

  const std::vector<float> &get(const std::string &name);
  std::vector<int> shape(const std::string &name) const;
};

class WeightLoader {
private:
  struct WeightEntry {
    std::string file;
    std::vector<int> shape;
    std::string dtype;
    std::vector<float> data;
    bool loaded;
  };

  std::string m_base_dir;
  std::unordered_map<std::string, WeightEntry> m_manifest;

  void parse_manifest(const std::string &json_path);
  void load_weight(const std::string &name);
  std::size_t compute_element_count(const std::vector<int> &shape) const;

public:
  WeightLoader() {}
  WeightLoader(const std::string &dir_path);

  const std::vector<float> &get(const std::string &name);
  std::vector<int> shape(const std::string &name) const;
  ScopedWeightLoader scope(const std::string &prefix);
  bool has(const std::string &name) const;
  void set_model_path(const std::string &dir_path);
};

} // namespace fgtm_tools

#endif
