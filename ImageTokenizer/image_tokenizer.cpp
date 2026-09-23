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
    unsigned char * raw_image = nullptr;
    raw_image = stbi_load(path_image.c_str(), &width, &height, &channels, m_channels);

    auto image_resized = resize224(raw_image, width, height);
    stbi_image_free(raw_image);
    unsigned char *image_data = image_resized.data();
    width = height = 224;

    std::printf("Image: %d x %d (channels: %d)\n", width, height, channels);

    int x_patches = width / m_patch_size;
    int y_patches = height / m_patch_size;
    int total_patches = x_patches * y_patches;
    int token_dim = m_patch_size * m_patch_size * m_channels;

    std::printf("Patches: %d x %d = %d total\n", x_patches, y_patches, total_patches);
    std::printf("Token dim: %d\n", token_dim);

    std::vector<float> tokens(total_patches * token_dim);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

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
                   
                    for (int c = 0; c < m_channels; c++)
                    {
                        float val = static_cast<float>(image_data[off_pixel + c]) / 255.0f;
                        token[idx++] = (val - mean[c]) / std[c];
                    }
                }
            }

        }
    }
    
    return tokens;
}

std::vector<unsigned char> ImageTokenizer::resize224(const unsigned char *src, int width, int height)
{
    std::vector<unsigned char> dst(224 * 224 * m_channels);
    float sx = static_cast<float>(width) / 224.0f;
    float sy = static_cast<float>(height) / 224.0f;

    for (int oy = 0; oy < 224; oy++)
    {
        for (int ox = 0; ox < 224; ox++)
        {
            float ix = (ox + 0.5f) * sx - 0.5f;
            float iy = (oy + 0.5f) * sy - 0.5f;
            int x0 = std::max(0, (int)ix);
            int y0 = std::max(0, (int)iy);
            int x1 = std::min(width - 1, x0 + 1);
            int y1 = std::min(height - 1, y0 + 1);
            float tx = ix - x0, ty = iy - y0;

            for (int c = 0; c < (int)m_channels; c++)
            {
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
}
