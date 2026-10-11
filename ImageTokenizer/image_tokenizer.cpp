#include "image_tokenizer.hpp"
#include "vendor/stb_image/stb_image.h"
#include <algorithm>
#include <cstdio>
#include <fstream>

namespace fgtm_tools {

ImageTokenizer::ImageTokenizer(std::size_t patch_size,
                               std::size_t embedding_dim, std::size_t channels)
    : m_patch_size(patch_size), m_embedding_dim(embedding_dim),
      m_channels(channels) {
  if (m_channels != 1 && m_channels != 3 && m_channels != 4) {
    throw std::invalid_argument(
        "Invalid number of channels. Supported values are 1, 3, or 4.");
  }
}

flib::ftensor ImageTokenizer::tokenize(const std::string &path_image,
                                       const flib::ftensor &projection_weight,
                                       const flib::ftensor &projection_bias,
                                       const flib::ftensor &cls_token,
                                       const flib::ftensor &position_embeddings,
                                       sycl::queue &queue) {
  return tokenize_image(path_image, projection_weight, projection_bias,
                        cls_token, position_embeddings, 224, queue);
}

flib::ftensor ImageTokenizer::tokenize512(
    const std::string &path_image, const flib::ftensor &projection_weight,
    const flib::ftensor &projection_bias, const flib::ftensor &cls_token,
    const flib::ftensor &position_embeddings, sycl::queue &queue) {
  return tokenize_image(path_image, projection_weight, projection_bias,
                        cls_token, position_embeddings, 512, queue);
}

flib::ftensor ImageTokenizer::tokenize_image(
    const std::string &path_image, const flib::ftensor &projection_weight,
    const flib::ftensor &projection_bias, const flib::ftensor &cls_token,
    const flib::ftensor &position_embeddings, std::size_t image_size,
    sycl::queue &queue) {
  int width, height, channels;
  std::vector<unsigned char> image_resized;
  if (image_size == 512) {
    auto prepared_image = load_and_prepare_rgba(path_image, width, height);
    channels = static_cast<int>(m_channels);
    image_resized = resize512(prepared_image.data(), width, height);
  } else {
    unsigned char *raw_image =
        stbi_load(path_image.c_str(), &width, &height, &channels, m_channels);
    if (raw_image == nullptr) {
      throw std::runtime_error("Failed to load image: " + path_image);
    }
    image_resized = resize224(raw_image, width, height);
    stbi_image_free(raw_image);
  }
  width = height = static_cast<int>(image_size);

  std::printf("Image: %d x %d (channels: %d)\n", width, height, channels);

  int x_patches = width / m_patch_size;
  int y_patches = height / m_patch_size;
  int total_patches = x_patches * y_patches;
  int token_dim = m_patch_size * m_patch_size * m_channels;

  std::printf("Patches: %d x %d = %d total\n", x_patches, y_patches,
              total_patches);
  std::printf("Token dim: %d\n", token_dim);

  std::vector<float> normalized_image =
      normalize_image(image_resized.data(), width, height);

  /*std::ofstream preprocessed_output("funlib_preprocessed_input.bin",
                                    std::ios::binary);
  if (!preprocessed_output.is_open()) {
    throw std::runtime_error("Failed to open preprocessed image output: "
                             "funlib_preprocessed_input.bin");
  }
  preprocessed_output.write(
      reinterpret_cast<const char *>(normalized_image.data()),
      static_cast<std::streamsize>(normalized_image.size() * sizeof(float)));
  if (!preprocessed_output) {
    throw std::runtime_error("Failed to write preprocessed image output: "
                             "funlib_preprocessed_input.bin");
  }*/

  std::vector<float> tokens =
      extract_normalized_patches(normalized_image.data(), width, height);

  flib::ftensor raw_tokens({static_cast<std::size_t>(total_patches),
                            static_cast<std::size_t>(token_dim)},
                           queue);
  raw_tokens.copy_from(tokens.data(), queue).wait();

  auto projected = flib::tensor_operations::gemm_batched(
      raw_tokens, projection_weight.transpose(), queue);

  auto projected_with_bias =
      flib::operations::add_bias(projected, projection_bias, queue);

  auto sequence = flib::tensor_operations::concatenate(
      cls_token, projected_with_bias, 0, queue);

  if (image_size == 512) {
    auto resized_positions =
        resize_position_embeddings(position_embeddings, queue);
    return flib::operations::add(sequence, resized_positions, queue);
  }

  return flib::operations::add(sequence, position_embeddings, queue);
}

void ImageTokenizer::save_rgb_ppm(
    const std::string &path, const std::vector<unsigned char> &pixels,
    int width, int height) const {
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to save preprocessed image: " + path);
  }

  output << "P6\n" << width << ' ' << height << "\n255\n";
  output.write(reinterpret_cast<const char *>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));
  if (!output) {
    throw std::runtime_error("Failed to write preprocessed image: " + path);
  }
}

std::vector<unsigned char>
ImageTokenizer::load_and_prepare_rgba(const std::string &path_image, int &width,
                                      int &height) const {
  int source_channels = 0;
  unsigned char *rgba =
      stbi_load(path_image.c_str(), &width, &height, &source_channels, 4);
  if (rgba == nullptr) {
    throw std::runtime_error("Failed to load image: " + path_image);
  }

  int min_x = width;
  int min_y = height;
  int max_x = -1;
  int max_y = -1;
  bool has_transparency = false;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const unsigned char alpha = rgba[(y * width + x) * 4 + 3];
      has_transparency = has_transparency || alpha < 255;
      if (alpha > 0) {
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
      }
    }
  }

  if (max_x < min_x || max_y < min_y) {
    stbi_image_free(rgba);
    throw std::runtime_error("Image contains no visible pixels: " +
                             path_image);
  }

  if (!has_transparency) {
    std::vector<unsigned char> rgb(width * height * m_channels);
    for (int pixel = 0; pixel < width * height; ++pixel) {
      for (int channel = 0; channel < static_cast<int>(m_channels); ++channel) {
        rgb[pixel * m_channels + channel] = rgba[pixel * 4 + channel];
      }
    }
    stbi_image_free(rgba);
    return rgb;
  }

  constexpr float foreground_ratio = 0.85f;
  const int crop_width = max_x - min_x + 1;
  const int crop_height = max_y - min_y + 1;
  const int foreground_size = std::max(crop_width, crop_height);
  const int framed_size =
      std::max(foreground_size,
               static_cast<int>(foreground_size / foreground_ratio));
  const int square_offset_x = (foreground_size - crop_width) / 2;
  const int square_offset_y = (foreground_size - crop_height) / 2;
  const int frame_offset = (framed_size - foreground_size) / 2;
  const int offset_x = frame_offset + square_offset_x;
  const int offset_y = frame_offset + square_offset_y;

  // TripoSR composites transparent pixels over 50% gray before encoding.
  std::vector<unsigned char> rgb(framed_size * framed_size * m_channels, 127);
  for (int y = 0; y < crop_height; ++y) {
    for (int x = 0; x < crop_width; ++x) {
      const int source = ((min_y + y) * width + min_x + x) * 4;
      const int destination =
          ((offset_y + y) * framed_size + offset_x + x) * m_channels;
      const float alpha = static_cast<float>(rgba[source + 3]) / 255.0f;

      for (int channel = 0; channel < static_cast<int>(m_channels); ++channel) {
        const float value = static_cast<float>(rgba[source + channel]) * alpha +
                            127.5f * (1.0f - alpha);
        rgb[destination + channel] = static_cast<unsigned char>(value);
      }
    }
  }

  stbi_image_free(rgba);
  width = height = framed_size;
  return rgb;
}

void ImageTokenizer::export_preprocessed512(const std::string &path_image,
                                             const std::string &path_output) {
  int width = 0;
  int height = 0;
  auto prepared_image = load_and_prepare_rgba(path_image, width, height);
  auto image_resized = resize512(prepared_image.data(), width, height);
  save_rgb_ppm(path_output, image_resized, 512, 512);
}

flib::ftensor ImageTokenizer::resize_position_embeddings(
    const flib::ftensor &position_embeddings, sycl::queue &queue) const {
  auto cls_position = flib::tensor_operations::slice(
      position_embeddings, 0, 0, 1, queue);
  auto patch_positions = flib::tensor_operations::slice(
      position_embeddings, 0, 1, 197, queue);

  // Interpolate only the learned 14x14 patch grid. The CLS position is kept.
  patch_positions.reshape({1, 14, 14, m_embedding_dim});
  patch_positions = flib::tensor_operations::permute(
      patch_positions, {0, 3, 1, 2}, queue);
  patch_positions =
      flib::operations::interpolate_bicubic(patch_positions, 32, 32, queue);
  patch_positions = flib::tensor_operations::permute(
      patch_positions, {0, 2, 3, 1}, queue);
  patch_positions.reshape({1024, m_embedding_dim});

  return flib::tensor_operations::concatenate(cls_position, patch_positions,
                                               0, queue);
}

std::vector<float>
ImageTokenizer::extract_patches(const unsigned char *image_data, int width,
                                int height) const {
  std::vector<float> normalized_image =
      normalize_image(image_data, width, height);
  return extract_normalized_patches(normalized_image.data(), width, height);
}

std::vector<float>
ImageTokenizer::normalize_image(const unsigned char *image_data, int width,
                                int height) const {
  std::vector<float> normalized_image(m_channels * width * height);
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float std[3] = {0.229f, 0.224f, 0.225f};

  for (int channel = 0; channel < static_cast<int>(m_channels); ++channel) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int source =
            (y * width + x) * static_cast<int>(m_channels) + channel;
        const int destination = channel * width * height + y * width + x;
        const float value = static_cast<float>(image_data[source]) / 255.0f;
        normalized_image[destination] = (value - mean[channel]) / std[channel];
      }
    }
  }

  return normalized_image;
}

std::vector<float>
ImageTokenizer::extract_normalized_patches(const float *normalized_image,
                                           int width, int height) const {
  const int x_patches = width / static_cast<int>(m_patch_size);
  const int y_patches = height / static_cast<int>(m_patch_size);
  const int token_dim =
      static_cast<int>(m_patch_size * m_patch_size * m_channels);
  std::vector<float> tokens(x_patches * y_patches * token_dim);

  for (int py = 0; py < y_patches; ++py) {
    for (int px = 0; px < x_patches; ++px) {
      float *token = tokens.data() + (py * x_patches + px) * token_dim;
      int index = 0;
      for (int channel = 0; channel < static_cast<int>(m_channels); ++channel) {
        for (int y = 0; y < static_cast<int>(m_patch_size); ++y) {
          for (int x = 0; x < static_cast<int>(m_patch_size); ++x) {
            const int image_x = px * static_cast<int>(m_patch_size) + x;
            const int image_y = py * static_cast<int>(m_patch_size) + y;
            const int pixel =
                channel * width * height + image_y * width + image_x;
            token[index++] = normalized_image[pixel];
          }
        }
      }
    }
  }

  return tokens;
}

std::vector<unsigned char> ImageTokenizer::resize224(const unsigned char *src,
                                                     int width, int height) {
  std::vector<unsigned char> dst(224 * 224 * m_channels);
  float sx = static_cast<float>(width) / 224.0f;
  float sy = static_cast<float>(height) / 224.0f;

  for (int oy = 0; oy < 224; oy++) {
    for (int ox = 0; ox < 224; ox++) {
      float ix = (ox + 0.5f) * sx - 0.5f;
      float iy = (oy + 0.5f) * sy - 0.5f;
      int x0 = std::max(0, (int)ix);
      int y0 = std::max(0, (int)iy);
      int x1 = std::min(width - 1, x0 + 1);
      int y1 = std::min(height - 1, y0 + 1);
      float tx = ix - x0, ty = iy - y0;

      for (int c = 0; c < (int)m_channels; c++) {
        float p00 = src[(y0 * width + x0) * m_channels + c];
        float p10 = src[(y0 * width + x1) * m_channels + c];
        float p01 = src[(y1 * width + x0) * m_channels + c];
        float p11 = src[(y1 * width + x1) * m_channels + c];
        dst[(oy * 224 + ox) * m_channels + c] = static_cast<unsigned char>(
            std::clamp((1 - tx) * (1 - ty) * p00 + tx * (1 - ty) * p10 +
                           (1 - tx) * ty * p01 + tx * ty * p11,
                       0.0f, 255.0f));
      }
    }
  }
  return dst;
}

std::vector<unsigned char> ImageTokenizer::resize512(const unsigned char *src,
                                                     int width, int height) {
  constexpr int output_size = 512;
  std::vector<unsigned char> dst(output_size * output_size * m_channels);
  const float sx = static_cast<float>(width) / output_size;
  const float sy = static_cast<float>(height) / output_size;

  for (int oy = 0; oy < output_size; ++oy) {
    for (int ox = 0; ox < output_size; ++ox) {
      const float ix = (ox + 0.5f) * sx - 0.5f;
      const float iy = (oy + 0.5f) * sy - 0.5f;
      const int x0 = std::max(0, static_cast<int>(ix));
      const int y0 = std::max(0, static_cast<int>(iy));
      const int x1 = std::min(width - 1, x0 + 1);
      const int y1 = std::min(height - 1, y0 + 1);
      const float tx = ix - x0;
      const float ty = iy - y0;

      for (int channel = 0; channel < static_cast<int>(m_channels);
           ++channel) {
        const float p00 =
            src[(y0 * width + x0) * m_channels + channel];
        const float p10 =
            src[(y0 * width + x1) * m_channels + channel];
        const float p01 =
            src[(y1 * width + x0) * m_channels + channel];
        const float p11 =
            src[(y1 * width + x1) * m_channels + channel];

        dst[(oy * output_size + ox) * m_channels + channel] =
            static_cast<unsigned char>(std::clamp(
                (1 - tx) * (1 - ty) * p00 + tx * (1 - ty) * p10 +
                    (1 - tx) * ty * p01 + tx * ty * p11,
                0.0f, 255.0f));
      }
    }
  }

  return dst;
}
} // namespace fgtm_tools
