#include "image_tokenizer.hpp"
#include "vendor/stb_image/stb_image.h"
#include <cstdio>

namespace fgtm_tools{

    ImageTokenizer::ImageTokenizer(std::size_t patch_size, std::size_t embedding_dim, std::size_t channels)
        : m_patch_size(patch_size), m_embedding_dim(embedding_dim), m_channels(channels)
    {
        if (m_channels != 1 && m_channels != 3 && m_channels != 4)
        {
            throw std::invalid_argument("Invalid number of channels. Supported values are 1, 3, or 4.");
        }
    }

    flib::ftensor ImageTokenizer::tokenize(
        const std::string &path_image,
        const flib::ftensor &projection_weight,
        const flib::ftensor &projection_bias,
        const flib::ftensor &cls_token,
        const flib::ftensor &position_embeddings,
        sycl::queue &queue)
    {
        int width, height, channels;
        unsigned char *raw_image = nullptr;
        raw_image = stbi_load(path_image.c_str(), &width, &height, &channels, m_channels);
        if (raw_image == nullptr)
        {
            throw std::runtime_error("Failed to load image: " + path_image);
        }

        auto image_resized = resize224(raw_image, width, height);
        stbi_image_free(raw_image);
        width = height = 224;

        std::printf("Image: %d x %d (channels: %d)\n", width, height, channels);

        int x_patches = width / m_patch_size;
        int y_patches = height / m_patch_size;
        int total_patches = x_patches * y_patches;
        int token_dim = m_patch_size * m_patch_size * m_channels;

        std::printf("Patches: %d x %d = %d total\n", x_patches, y_patches, total_patches);
        std::printf("Token dim: %d\n", token_dim);

        std::vector<float> tokens =
            extract_patches(image_resized.data(), width, height);

        flib::ftensor raw_tokens({static_cast<std::size_t>(total_patches),
                                  static_cast<std::size_t>(token_dim)},
                                 queue);
        raw_tokens.copy_from(tokens.data(), queue).wait();

        auto projected = flib::tensor_operations::gemm_batched(
            raw_tokens, projection_weight, queue, false, true);

        auto projected_with_bias = flib::operations::add_bias(
            projected, projection_bias, queue);

        flib::ftensor sequence({static_cast<std::size_t>(total_patches + 1),
                                m_embedding_dim},
                               queue);
        const float *projected_data = projected_with_bias.device_data();
        const float *cls_data = cls_token.device_data();
        float *sequence_data = sequence.device_data();
        const std::size_t embedding_dim = m_embedding_dim;
        const std::size_t sequence_size =
            static_cast<std::size_t>(total_patches + 1) * embedding_dim;

        queue.parallel_for(sycl::range<1>{sequence_size}, [=](sycl::id<1> id) {
            const std::size_t index = id[0];
            if (index < embedding_dim)
            {
                sequence_data[index] = cls_data[index];
            }
            else
            {
                sequence_data[index] = projected_data[index - embedding_dim];
            }
        }).wait();

        return flib::operations::add(
            sequence, position_embeddings, queue);
    }

    std::vector<float> ImageTokenizer::extract_patches(
        const unsigned char *image_data, int width, int height) const
    {
        const int x_patches = width / static_cast<int>(m_patch_size);
        const int y_patches = height / static_cast<int>(m_patch_size);
        const int token_dim = static_cast<int>(m_patch_size * m_patch_size * m_channels);
        std::vector<float> tokens(x_patches * y_patches * token_dim);
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};

        for (int py = 0; py < y_patches; ++py)
        {
            for (int px = 0; px < x_patches; ++px)
            {
                float *token = tokens.data() + (py * x_patches + px) * token_dim;
                int index = 0;
                for (int channel = 0; channel < static_cast<int>(m_channels); ++channel)
                {
                    for (int y = 0; y < static_cast<int>(m_patch_size); ++y)
                    {
                        for (int x = 0; x < static_cast<int>(m_patch_size); ++x)
                        {
                            const int image_x = px * static_cast<int>(m_patch_size) + x;
                            const int image_y = py * static_cast<int>(m_patch_size) + y;
                            const int pixel = (image_y * width + image_x) *
                                              static_cast<int>(m_channels) + channel;
                            const float value =
                                static_cast<float>(image_data[pixel]) / 255.0f;
                            token[index++] = (value - mean[channel]) / std[channel];
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
