#include "fungt_mesher.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

flib::ftensor upload_tensor(const std::vector<std::size_t> &shape,
                            const std::vector<float> &values,
                            sycl::queue &queue) {
  flib::ftensor tensor(shape, queue);
  tensor.copy_from(values.data(), queue).wait();
  return tensor;
}

} // namespace

FunGTMesher::FunGTMesher() {

  m_image_tokenizer = std::make_shared<fgtm_tools::ImageTokenizer>();
  m_image_encoder = std::make_shared<fgtm_tools::ImageEncoder>();

  switch (ComputeBackend::GetBackend()) {
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
                                     m_vendor, m_backend, true);
  m_queue = flib::sycl_handler::get_queue(m_backend_name);
  flib::sycl_handler::get_device_info(m_backend_name);
}

void FunGTMesher::generate(const std::string &path_image,
                           const std::string &path_output) {
  if (!m_model_set) {
    throw std::runtime_error(
        "FunGTMesher::set_model() must be called before generate()");
  }

  const std::filesystem::path output_path(path_output);
  const std::filesystem::path diagnostic_directory =
      output_path.has_parent_path() ? output_path.parent_path() : ".";

  auto image_features = image_tokenizer(path_image);
  export_image_features(
      image_features,
      (diagnostic_directory / "funlib_after_embeddings.bin").string());

  image_encoder(image_features, diagnostic_directory.string());
  export_image_features(image_features, path_output);
}

void FunGTMesher::export_image_features(const flib::ftensor &features,
                                        const std::string &path_output) {
  auto host_features = features.to_host(m_queue);
  std::ofstream output(path_output, std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open image feature output: " +
                             path_output);
  }

  output.write(
      reinterpret_cast<const char *>(host_features.data()),
      static_cast<std::streamsize>(host_features.size() * sizeof(float)));
  if (!output) {
    throw std::runtime_error("Failed to write image features: " + path_output);
  }
}

flib::ftensor FunGTMesher::image_tokenizer(const std::string &path_image) {
  auto weights = m_weight_loader.scope("image_tokenizer");

  const auto &weight_data = weights.get("patch_projection.weight");

  const auto &bias_data = weights.get("patch_projection.bias");

  const auto &cls_token_data = weights.get("cls_token");

  const auto &pos_embed_data = weights.get("position_embeddings");

  flib::ftensor projection_weight({768, 768}, m_queue);
  flib::ftensor projection_bias({768}, m_queue);
  flib::ftensor cls_token({1, 768}, m_queue);
  flib::ftensor position_embeddings({197, 768}, m_queue);

  projection_weight.copy_from(weight_data.data(), m_queue).wait();
  projection_bias.copy_from(bias_data.data(), m_queue).wait();
  cls_token.copy_from(cls_token_data.data(), m_queue).wait();
  position_embeddings.copy_from(pos_embed_data.data(), m_queue).wait();

  auto tokens = m_image_tokenizer->tokenize(path_image, projection_weight,
                                            projection_bias, cls_token,
                                            position_embeddings, m_queue);

  return tokens;
}

void FunGTMesher::image_encoder(
    flib::ftensor &tokens, const std::string &diagnostic_directory) {
  for (std::size_t layer = 0; layer < 12; ++layer) {
    auto weights = m_weight_loader.scope("image_tokenizer");
    const std::string layer_prefix =
        "encoder.layer." + std::to_string(layer) + ".";

    fgtm_tools::EncoderLayerWeights device_weights;
    device_weights.norm_before_weight =
        upload_tensor({768}, weights.get(layer_prefix + "layernorm_before.weight"), m_queue);
    device_weights.norm_before_bias =
        upload_tensor({768}, weights.get(layer_prefix + "layernorm_before.bias"), m_queue);

    device_weights.query_weight = upload_tensor(
        {768, 768}, weights.get(layer_prefix + "attention.attention.query.weight"), m_queue);
    device_weights.query_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.attention.query.bias"), m_queue);
    device_weights.key_weight = upload_tensor(
        {768, 768}, weights.get(layer_prefix + "attention.attention.key.weight"), m_queue);
    device_weights.key_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.attention.key.bias"), m_queue);
    device_weights.value_weight = upload_tensor(
        {768, 768}, weights.get(layer_prefix + "attention.attention.value.weight"), m_queue);
    device_weights.value_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.attention.value.bias"), m_queue);

    device_weights.attention_weight = upload_tensor(
        {768, 768}, weights.get(layer_prefix + "attention.output.dense.weight"), m_queue);
    device_weights.attention_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.output.dense.bias"), m_queue);

    device_weights.norm_after_weight =
        upload_tensor({768}, weights.get(layer_prefix + "layernorm_after.weight"), m_queue);
    device_weights.norm_after_bias =
        upload_tensor({768}, weights.get(layer_prefix + "layernorm_after.bias"), m_queue);

    device_weights.ffn_expand_weight = upload_tensor(
        {3072, 768}, weights.get(layer_prefix + "intermediate.dense.weight"), m_queue);
    device_weights.ffn_expand_bias =
        upload_tensor({3072}, weights.get(layer_prefix + "intermediate.dense.bias"), m_queue);
    device_weights.ffn_reduce_weight =
        upload_tensor({768, 3072}, weights.get(layer_prefix + "output.dense.weight"), m_queue);
    device_weights.ffn_reduce_bias =
        upload_tensor({768}, weights.get(layer_prefix + "output.dense.bias"), m_queue);

    m_image_encoder->set_weights(std::move(device_weights));
    m_image_encoder->encode_layer(tokens, m_queue);

    export_image_features(
        tokens,
        (std::filesystem::path(diagnostic_directory) /
         ("funlib_after_layer" + std::to_string(layer) + ".bin"))
            .string());
  }

  auto final_norm_weight = upload_tensor(
      {768},
      m_weight_loader.get("image_tokenizer.model.layernorm.weight"),
      m_queue);
  auto final_norm_bias = upload_tensor(
      {768},
      m_weight_loader.get("image_tokenizer.model.layernorm.bias"),
      m_queue);

  tokens = flib::operations::layer_norm(
      tokens, final_norm_weight, final_norm_bias, 1.0e-12f, m_queue);
}
