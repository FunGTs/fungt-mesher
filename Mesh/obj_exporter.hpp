#if !defined(_OBJ_EXPORTER_HPP_)
#define _OBJ_EXPORTER_HPP_

#include <funlib/graphics/marching_cubes.hpp>
#include <string>

namespace fgtm {

class ObjExporter {
public:
  void export_mesh(
      const flib::graphics::marching_cubes::Mesh &mesh,
      const std::string &path,
      sycl::queue &queue) const;
};

} // namespace fgtm

#endif // _OBJ_EXPORTER_HPP_
