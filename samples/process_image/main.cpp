#include "ImageTokenizer/image_tokenizer.hpp"

#include <exception>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path> <output.ppm>\n";
    return 1;
  }

  try {
    fgtm_tools::ImageTokenizer tokenizer;
    tokenizer.export_preprocessed512(argv[1], argv[2]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
