#include "image_tokenizer.hpp"
#include "vendor/stb_image/stb_image.h"
#include <cstdio>

namespace fungt::mesher {

ImageTokenizer::ImageTokenizer(std::size_t patch_size, std::size_t embedding_dim, std::size_t channels)
: m_patch_size(patch_size), m_embedding_dim(embedding_dim), m_channels(channels)
{
    if (m_channels != 1 && m_channels != 3 && m_channels != 4) {
        throw std::invalid_argument("Invalid number of channels. Supported values are 1, 3, or 4.");
    }
}

std::vector<float> ImageTokenizer::tokenize(const std::string &path_image)
{
    int width, height, channels;
    unsigned char * image_data = stbi_load(path_image.c_str(), &width, &height, &channels, m_channels);

    std::printf("Image: %d x %d (channels: %d)\n", width, height, channels);

    int x_patches = width / m_patch_size;
    int y_patches = height / m_patch_size;
    int total_patches = x_patches * y_patches;
    int token_dim = m_patch_size * m_patch_size * m_channels;

    std::printf("Patches: %d x %d = %d total\n", x_patches, y_patches, total_patches);
    std::printf("Token dim: %d\n", token_dim);

    std::vector<float> tokens(total_patches * token_dim);

    for(int py = 0; py < y_patches; py++){
        for(int px = 0; px < x_patches; px++){

            int id_patch = py*x_patches + px;
            float* token = tokens.data() + id_patch * token_dim;
            int idx = 0;

            for(int y = 0; y < m_patch_size; y++){
                for(int x = 0; x < m_patch_size; x++){

                    int image_x = px*m_patch_size + x;
                    int image_y = py*m_patch_size + y;
                    int off_pixel = (image_y*width + image_x)*m_channels;

                    for(int c = 0; c < m_channels; c++){
                        token[idx++] = static_cast<float>(image_data[off_pixel + c]) / 255.0f;
                    }

                }
            }

        }
    }


    stbi_image_free(image_data);

    return tokens;
}

}
