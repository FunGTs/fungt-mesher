#if !defined(_IMAGE_TOKENIZER_HPP_)
#define _IMAGE_TOKENIZER_HPP_
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <funlib/funlib.hpp>
namespace fungt::mesher {

    class ImageTokenizer
    {

    private:
        std::size_t m_patch_size;
        std::size_t m_embedding_dim;
        std::size_t m_channels;
        std::size_t m_in_features;

    public:
        ImageTokenizer(std::size_t patch_size = 16, std::size_t embedding_dim = 768, std::size_t channels = 3);
        std::vector<float> tokenize(const std::string &path_image);
    };
}

#endif // _IMAGE_TOKENIZER_HPP_
