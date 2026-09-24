#include "image_tokenizer.hpp"
#include <cstdio>
#include <vector>
#include <cmath>

int main()
{
    std::printf("=== ImageTokenizer Normalization Test ===\n\n");

    std::vector<unsigned char> image(224 * 224 * 3, 100);
    fgtm_tools::ImageTokenizer tokenizer(16, 768, 3);
    std::vector<float> tokens = tokenizer.extract_patches(image.data(), 224, 224);

    std::printf("Expected for grey (100,100,100): approx (-0.41, -0.29, -0.06)\n");
    std::printf("Actual RGB:                      (%.2f, %.2f, %.2f)\n\n",
                tokens[0], tokens[256], tokens[512]);

    return 0;
}
