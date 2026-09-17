#include <iostream>
#include "image_tokenizer.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    fungt::mesher::ImageTokenizer tokenizer(16, 768, 3);

    auto tokens = tokenizer.tokenize(argv[1]);

    int token_dim = 16 * 16 * 3;
    std::cout << "Total floats: " << tokens.size() << std::endl;
    std::cout << "Total tokens: " << tokens.size() / token_dim << std::endl;

    return 0;
}
