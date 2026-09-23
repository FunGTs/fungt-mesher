#include "image_tokenizer.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
#include <fstream>

void create_grey_test_image(const std::string& path)
{
    const int width = 224;
    const int height = 224;
    const unsigned char grey = 100;

    std::vector<unsigned char> header = {
        0x42, 0x4D,
        0x36, 0x00, 0x0C, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x36, 0x00, 0x00, 0x00,
        0x28, 0x00, 0x00, 0x00,
        0xE0, 0x00, 0x00, 0x00,
        0xE0, 0x00, 0x00, 0x00,
        0x01, 0x00,
        0x18, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x0C, 0x00,
        0x13, 0x0B, 0x00, 0x00,
        0x13, 0x0B, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    };

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<char*>(header.data()), header.size());

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            file.put(grey);
            file.put(grey);
            file.put(grey);
        }
    }
    file.close();
}

int main()
{
    std::printf("=== ImageTokenizer Normalization Test ===\n\n");

    const std::string test_image = "grey_test.bmp";
    create_grey_test_image(test_image);
    std::printf("Created grey test image (100, 100, 100)\n\n");

    fungt::mesher::ImageTokenizer tokenizer(16, 768, 3);
    std::vector<float> tokens = tokenizer.tokenize(test_image);

    std::printf("First token's first 6 values (R,G,B,R,G,B):\n");
    std::printf("  [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n\n",
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]);

    std::printf("Expected for grey (100,100,100): approx (-0.41, -0.29, -0.06)\n");
    std::printf("Actual RGB:                      (%.2f, %.2f, %.2f)\n\n",
                tokens[0], tokens[1], tokens[2]);

    std::remove(test_image.c_str());

    return 0;
}
