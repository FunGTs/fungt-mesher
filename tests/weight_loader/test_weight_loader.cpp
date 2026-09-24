#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "WeightLoader/weight_loader.hpp"

bool test_cls_token(fgtm_tools::WeightLoader& loader) {
    std::printf("Testing cls_token...\n");

    auto scoped = loader.scope("image_tokenizer");
    const auto& cls = scoped.get("cls_token");

    if (cls.size() != 768) {
        std::printf("  FAIL: expected 768 floats, got %zu\n", cls.size());
        return false;
    }

    auto shape = scoped.shape("cls_token");
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 768) {
        std::printf("  FAIL: unexpected shape\n");
        return false;
    }

    std::printf("  PASS: cls_token has %zu floats\n", cls.size());
    return true;
}

bool test_patch_projection_weight(fgtm_tools::WeightLoader& loader) {
    std::printf("Testing patch_projection.weight...\n");

    auto scoped = loader.scope("image_tokenizer");
    const auto& proj = scoped.get("patch_projection.weight");

    if (proj.size() != 589824) {
        std::printf("  FAIL: expected 589824 floats (768x3x16x16), got %zu\n",
                    proj.size());
        return false;
    }

    auto shape = scoped.shape("patch_projection.weight");
    int expected_total = 768 * 3 * 16 * 16;
    if (shape.size() != 4 || shape[0] != 768 || shape[1] != 3 ||
        shape[2] != 16 || shape[3] != 16) {
        std::printf("  FAIL: unexpected shape\n");
        return false;
    }

    std::printf("  PASS: patch_projection.weight has %zu floats\n", proj.size());
    return true;
}

bool test_position_embeddings(fgtm_tools::WeightLoader& loader) {
    std::printf("Testing position_embeddings...\n");

    auto scoped = loader.scope("image_tokenizer");
    const auto& pos = scoped.get("position_embeddings");

    if (pos.size() != 151296) {
        std::printf("  FAIL: expected 151296 floats (197x768), got %zu\n",
                    pos.size());
        return false;
    }

    auto shape = scoped.shape("position_embeddings");
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 197 || shape[2] != 768) {
        std::printf("  FAIL: unexpected shape\n");
        return false;
    }

    std::printf("  PASS: position_embeddings has %zu floats\n", pos.size());
    return true;
}

bool test_scoped_loader(fgtm_tools::WeightLoader& loader) {
    std::printf("Testing ScopedWeightLoader...\n");

    auto scoped = loader.scope("image_tokenizer");

    const auto& direct = loader.get("image_tokenizer/cls_token");
    const auto& via_scope = scoped.get("cls_token");

    if (&direct != &via_scope) {
        std::printf("  FAIL: scoped and direct access return different data\n");
        return false;
    }

    std::printf("  PASS: scoped loader returns same data as direct access\n");
    return true;
}

bool test_not_found(fgtm_tools::WeightLoader& loader) {
    std::printf("Testing error on missing weight...\n");

    try {
        loader.get("nonexistent/weight");
        std::printf("  FAIL: expected exception for missing weight\n");
        return false;
    } catch (const std::runtime_error& e) {
        std::printf("  PASS: correctly threw exception\n");
        return true;
    }
}

bool test_lazy_loading(fgtm_tools::WeightLoader& loader) {
    std::printf("Testing lazy loading...\n");

    auto shape = loader.shape("image_tokenizer/cls_token");

    std::printf("  PASS: shape() returns without loading data\n");
    return true;
}

int main(int argc, char* argv[]) {
    const char* weights_dir = nullptr;

    if (argc > 1) {
        weights_dir = argv[1];
    } else {
        const char* env_dir = std::getenv("TRIPOSR_WEIGHTS_DIR");
        if (env_dir) {
            weights_dir = env_dir;
        }
    }

    if (!weights_dir) {
        std::printf("Usage: %s <path_to_triposr_weights>\n", argv[0]);
        std::printf("Or set TRIPOSR_WEIGHTS_DIR environment variable\n");
        return 1;
    }

    std::printf("=== WeightLoader Test Suite ===\n");
    std::printf("Weights directory: %s\n\n", weights_dir);

    try {
        fgtm_tools::WeightLoader loader(weights_dir);

        int passed = 0;
        int failed = 0;

        if (test_cls_token(loader)) ++passed; else ++failed;
        if (test_patch_projection_weight(loader)) ++passed; else ++failed;
        if (test_position_embeddings(loader)) ++passed; else ++failed;
        if (test_scoped_loader(loader)) ++passed; else ++failed;
        if (test_not_found(loader)) ++passed; else ++failed;
        if (test_lazy_loading(loader)) ++passed; else ++failed;

        std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);

        return failed > 0 ? 1 : 0;

    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 1;
    }
}
