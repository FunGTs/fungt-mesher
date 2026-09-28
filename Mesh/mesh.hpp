#if !defined(_MESH_HPP_)
#define _MESH_HPP_

namespace fgtm {
class Mesh {
public:
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<std::uint32_t> indices;

  void save_obj(const std::string &path) const;
};
}; // namespace fgtm

#endif // _MESH_HPP_
