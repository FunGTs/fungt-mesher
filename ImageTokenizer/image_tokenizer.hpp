#if !defined(_IMAGE_TOKENIZER_HPP_)
#define _IMAGE_TOKENIZER_HPP_
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <funlib/funlib.hpp>
namespace fgtm_tools{

    class ImageTokenizer
    {

    private:
        std::size_t m_patch_size;
        std::size_t m_embedding_dim;
        std::size_t m_channels;
        std::size_t m_in_features;

    public:
        ImageTokenizer(std::size_t patch_size = 16, std::size_t embedding_dim = 768, std::size_t channels = 3);
        flib::ftensor tokenize(const std::string &path_image,
                               const flib::ftensor &projection_weight,
                               const flib::ftensor &projection_bias,
                               const flib::ftensor &cls_token,
                               const flib::ftensor &position_embeddings,
                               sycl::queue &queue);
        std::vector<unsigned char> resize224(const unsigned char *src, int width, int height);
        std::vector<float> extract_patches(const unsigned char *image_data,
                                           int width, int height) const;
        void setup(std::size_t patch_size = 16, std::size_t embedding_dim = 768, std::size_t channels = 3);
    };
}

#endif // _IMAGE_TOKENIZER_HPP_
