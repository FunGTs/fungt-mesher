#if !defined(_IMAGE_TOKENIZER_HPP_)
#define _IMAGE_TOKENIZER_HPP_
#include <cmath>
#include <funlib/funlib.hpp>
#include <random>
#include <stdexcept>
#include <vector>
namespace fgtm_tools {

class ImageTokenizer {

private:
  std::size_t m_patch_size;
  std::size_t m_embedding_dim;
  std::size_t m_channels;
  std::size_t m_in_features;

  std::vector<float> normalize_image(const unsigned char *image_data, int width,
                                     int height) const;
  std::vector<float> extract_normalized_patches(const float *normalized_image,
                                                int width, int height) const;
  void save_rgb_ppm(const std::string &path,
                    const std::vector<unsigned char> &pixels, int width,
                    int height) const;
  std::vector<unsigned char>
  load_and_prepare_rgba(const std::string &path_image, int &width,
                        int &height) const;
  flib::ftensor resize_position_embeddings(
      const flib::ftensor &position_embeddings, sycl::queue &queue) const;
  flib::ftensor tokenize_image(const std::string &path_image,
                               const flib::ftensor &projection_weight,
                               const flib::ftensor &projection_bias,
                               const flib::ftensor &cls_token,
                               const flib::ftensor &position_embeddings,
                               std::size_t image_size, sycl::queue &queue);

public:
  ImageTokenizer(std::size_t patch_size = 16, std::size_t embedding_dim = 768,
                 std::size_t channels = 3);
  flib::ftensor tokenize(const std::string &path_image,
                         const flib::ftensor &projection_weight,
                         const flib::ftensor &projection_bias,
                         const flib::ftensor &cls_token,
                         const flib::ftensor &position_embeddings,
                         sycl::queue &queue);
  flib::ftensor tokenize512(const std::string &path_image,
                            const flib::ftensor &projection_weight,
                            const flib::ftensor &projection_bias,
                            const flib::ftensor &cls_token,
                            const flib::ftensor &position_embeddings,
                            sycl::queue &queue);
  std::vector<unsigned char> resize224(const unsigned char *src, int width,
                                       int height);
  std::vector<unsigned char> resize512(const unsigned char *src, int width,
                                       int height);
  void export_preprocessed512(const std::string &path_image,
                              const std::string &path_output);
  std::vector<float> extract_patches(const unsigned char *image_data, int width,
                                     int height) const;
  void setup(std::size_t patch_size = 16, std::size_t embedding_dim = 768,
             std::size_t channels = 3);
};
} // namespace fgtm_tools

#endif // _IMAGE_TOKENIZER_HPP_
