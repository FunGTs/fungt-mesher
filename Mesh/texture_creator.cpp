#include "texture_creator.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image/stb_image_write.h"

namespace fgtm {

TextureCoordinates TextureCreator::create_texture(
    const flib::ftensor &colors,
    const std::string &path,
    sycl::queue &queue) const {
  if (colors.getRank() != 2 || colors.getShape()[1] != 3) {
    throw std::invalid_argument("Color tensor must have shape [N, 3]");
  }

  const auto color_values = colors.to_host(queue);
  const std::size_t vertex_count = colors.getShape()[0];
  const std::size_t texture_width = 1024;
  const std::size_t texture_height =
      (vertex_count + texture_width - 1) / texture_width;
  std::vector<unsigned char> texture_pixels(
      texture_width * texture_height * 3, 0);
  TextureCoordinates coordinates{
      texture_width,
      texture_height,
      std::vector<float>(vertex_count * 2)};

  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    const std::size_t pixel = vertex * 3;
    texture_pixels[pixel] = static_cast<unsigned char>(
        std::clamp(color_values[pixel], 0.0f, 1.0f) * 255.0f);
    texture_pixels[pixel + 1] = static_cast<unsigned char>(
        std::clamp(color_values[pixel + 1], 0.0f, 1.0f) * 255.0f);
    texture_pixels[pixel + 2] = static_cast<unsigned char>(
        std::clamp(color_values[pixel + 2], 0.0f, 1.0f) * 255.0f);

    const std::size_t x = vertex % texture_width;
    const std::size_t y = vertex / texture_width;
    coordinates.values[vertex * 2] =
        (static_cast<float>(x) + 0.5f) /
        static_cast<float>(texture_width);
    coordinates.values[vertex * 2 + 1] =
        1.0f - (static_cast<float>(y) + 0.5f) /
                    static_cast<float>(texture_height);
  }

  if (stbi_write_png(
          path.c_str(),
          static_cast<int>(texture_width),
          static_cast<int>(texture_height),
          3,
          texture_pixels.data(),
          static_cast<int>(texture_width * 3)) == 0) {
    throw std::runtime_error("Failed to write color texture: " + path);
  }

  return coordinates;
}

} // namespace fgtm
