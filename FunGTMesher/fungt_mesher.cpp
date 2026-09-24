#include "fungt_mesher.hpp"

FunGTMesher::FunGTMesher(){

    m_image_tokenizer = std::make_shared<fgtm_tools::ImageTokenizer>();

    switch (ComputeBackend::GetBackend())
    {
    case Backend::SYCL_CUDA:
            m_backend_name = "cuda";
            m_vendor = flib::vendor::NVIDIA;
            m_backend = flib::backend::CUDA;
        break;
    case Backend::SYCL_LEVEL_ZERO:
            m_backend_name = "level_zero";
            m_vendor = flib::vendor::INTEL;
            m_backend = flib::backend::LEVEL_ZERO;
        break;
    case Backend::SYCL_OPENCL:
            m_backend_name = "opencl";
            m_vendor = flib::vendor::INTEL;
            m_backend = flib::backend::OPENCL;
        break;
    default:
        break;
    }

    flib::sycl_handler::register_queue(m_backend_name, flib::device::GPU,
                                       m_vendor, m_backend,
                                       true);
    m_queue = flib::sycl_handler::get_queue(m_backend_name);
    flib::sycl_handler::get_device_info(m_backend_name);
}

void FunGTMesher::generate(const std::string &path_image, const std::string &path_output)
{
    if (!m_model_set)
    {
        throw std::runtime_error(
            "FunGTMesher::set_model() must be called before generate()");
    }

    auto image_features = image_tokenizer(path_image);
}

flib::ftensor FunGTMesher::image_tokenizer(const std::string &path_image)
{
    auto weights = m_weight_loader.scope("image_tokenizer");

    const auto &weight_data =
        weights.get("patch_projection.weight");

    const auto &bias_data =
        weights.get("patch_projection.bias");

    const auto &cls_token_data =
        weights.get("cls_token");

    const auto &position_embeddings_data =
        weights.get("position_embeddings");

    flib::ftensor projection_weight({768, 768}, m_queue);
    flib::ftensor projection_bias({768}, m_queue);
    flib::ftensor cls_token({768}, m_queue);
    flib::ftensor position_embeddings({197, 768}, m_queue);

    projection_weight.copy_from(weight_data.data(), m_queue).wait();
    projection_bias.copy_from(bias_data.data(), m_queue).wait();
    cls_token.copy_from(cls_token_data.data(), m_queue).wait();
    position_embeddings.copy_from(position_embeddings_data.data(), m_queue).wait();

    auto tokens = m_image_tokenizer->tokenize(
        path_image,
        projection_weight,
        projection_bias,
        cls_token,
        position_embeddings,
        m_queue);

    return tokens;
}
