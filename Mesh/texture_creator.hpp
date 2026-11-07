#if !defined(_TEXTURE_CREATOR_HPP_)
#define _TEXTURE_CREATOR_HPP_

#include <funlib/Tensor/tensor.hpp>
#include <string>
#include <vector>

namespace fgtm {

struct TextureCoordinates {
  std::size_t width;
  std::size_t height;
  std::vector<float> values;
};

class TextureCreator {
public:
  TextureCoordinates create_texture(
      const flib::ftensor &colors,
      const std::string &path,
      sycl::queue &queue) const;
};

} // namespace fgtm

#endif // _TEXTURE_CREATOR_HPP_
