#include "fungt_mesher.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  m_triplane_decoder = std::make_shared<fgtm_tools::TriplaneDecoder>();

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

  using Clock = std::chrono::steady_clock;
  const auto total_start = Clock::now();

  const auto tokenizer_start = Clock::now();
  auto image_features = image_tokenizer(path_image);
  m_queue.wait_and_throw();
  const auto tokenizer_end = Clock::now();
  export_image_features(
      image_features,
      (diagnostic_directory / "funlib_after_embeddings.bin").string());

  const auto encoder_start = Clock::now();
  image_encoder(image_features, diagnostic_directory.string());
  m_queue.wait_and_throw();
  const auto encoder_end = Clock::now();

  // Remove CLS before cross-attention: [197, 768] -> [1, 196, 768].
  const auto cls_start = Clock::now();
  auto image_tokens = drop_cls_token(image_features);
  m_queue.wait_and_throw();
  const auto cls_end = Clock::now();

  // Convert encoded image tokens into triplane features.
  const auto decoder_start = Clock::now();
  auto triplane_features =
      decode_triplane(image_tokens, diagnostic_directory.string());
  m_queue.wait_and_throw();
  const auto decoder_end = Clock::now();
  export_image_features(triplane_features, path_output);

  const auto total_end = Clock::now();
  const auto milliseconds = [](Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  };

  std::cout << "Tokenizer: " << milliseconds(tokenizer_start, tokenizer_end)
            << " ms\n";
  std::cout << "Image encoder: " << milliseconds(encoder_start, encoder_end)
            << " ms\n";
  std::cout << "CLS removal: " << milliseconds(cls_start, cls_end) << " ms\n";
  std::cout << "Triplane decoder: " << milliseconds(decoder_start, decoder_end)
            << " ms\n";
  std::cout << "Total execution: " << milliseconds(total_start, total_end)
            << " ms\n";
}

flib::ftensor FunGTMesher::drop_cls_token(const flib::ftensor &tokens) {
  constexpr std::size_t token_count = 197;
  constexpr std::size_t embedding_size = 768;

  if (tokens.getShape() !=
      std::vector<std::size_t>{token_count, embedding_size}) {
    throw std::invalid_argument(
        "CLS removal expects image tokens with shape [197,768]");
  }

  // Funlib has no tensor slice operation yet, so copy all patch tokens after
  // CLS to the host and upload them as [B, N, D] = [1, 196, 768].
  const auto encoded = tokens.to_host(m_queue);
  std::vector<float> patches(encoded.begin() + embedding_size, encoded.end());
  return upload_tensor({1, token_count - 1, embedding_size}, patches, m_queue);
}

DecoderWeights FunGTMesher::load_decoder_weights() {
  auto weights = m_weight_loader.scope("decoder");
  DecoderWeights decoder;

  // These weights are used once before and after all 16 decoder blocks.
  decoder.norm = upload_tensor({1024}, weights.get("norm.weight"), m_queue);
  decoder.norm_bias = upload_tensor({1024}, weights.get("norm.bias"), m_queue);
  decoder.projection_in =
      upload_tensor({1024, 1024}, weights.get("proj_in.weight"), m_queue);
  decoder.projection_in_bias =
      upload_tensor({1024}, weights.get("proj_in.bias"), m_queue);
  decoder.projection_out =
      upload_tensor({1024, 1024}, weights.get("proj_out.weight"), m_queue);
  decoder.projection_out_bias =
      upload_tensor({1024}, weights.get("proj_out.bias"), m_queue);

  decoder.blocks.reserve(16);
  for (std::size_t block = 0; block < 16; ++block) {
    const std::string prefix = "block." + std::to_string(block) + ".";
    BlockWeights block_weights;

    // Self-attention weights for this block.
    block_weights.norm1 =
        upload_tensor({1024}, weights.get(prefix + "norm1.weight"), m_queue);
    block_weights.norm1_bias =
        upload_tensor({1024}, weights.get(prefix + "norm1.bias"), m_queue);
    block_weights.self_query = upload_tensor(
        {1024, 1024}, weights.get(prefix + "attn1.to_q.weight"), m_queue);
    block_weights.self_key = upload_tensor(
        {1024, 1024}, weights.get(prefix + "attn1.to_k.weight"), m_queue);
    block_weights.self_value = upload_tensor(
        {1024, 1024}, weights.get(prefix + "attn1.to_v.weight"), m_queue);
    block_weights.self_output = upload_tensor(
        {1024, 1024}, weights.get(prefix + "attn1.to_out.0.weight"), m_queue);
    block_weights.self_output_bias = upload_tensor(
        {1024}, weights.get(prefix + "attn1.to_out.0.bias"), m_queue);

    // Cross-attention projects image features from 768 to 1024.
    block_weights.norm2 =
        upload_tensor({1024}, weights.get(prefix + "norm2.weight"), m_queue);
    block_weights.norm2_bias =
        upload_tensor({1024}, weights.get(prefix + "norm2.bias"), m_queue);
    block_weights.cross_query = upload_tensor(
        {1024, 1024}, weights.get(prefix + "attn2.to_q.weight"), m_queue);
    block_weights.cross_key = upload_tensor(
        {1024, 768}, weights.get(prefix + "attn2.to_k.weight"), m_queue);
    block_weights.cross_value = upload_tensor(
        {1024, 768}, weights.get(prefix + "attn2.to_v.weight"), m_queue);
    block_weights.cross_output = upload_tensor(
        {1024, 1024}, weights.get(prefix + "attn2.to_out.0.weight"), m_queue);
    block_weights.cross_output_bias = upload_tensor(
        {1024}, weights.get(prefix + "attn2.to_out.0.bias"), m_queue);

    // GEGLU expands 1024 to 8192, halves it to 4096, then returns to 1024.
    block_weights.norm3 =
        upload_tensor({1024}, weights.get(prefix + "norm3.weight"), m_queue);
    block_weights.norm3_bias =
        upload_tensor({1024}, weights.get(prefix + "norm3.bias"), m_queue);
    block_weights.geglu = upload_tensor(
        {8192, 1024}, weights.get(prefix + "ff.net.0.proj.weight"), m_queue);
    block_weights.geglu_bias = upload_tensor(
        {8192}, weights.get(prefix + "ff.net.0.proj.bias"), m_queue);
    block_weights.ffn_output = upload_tensor(
        {1024, 4096}, weights.get(prefix + "ff.net.2.weight"), m_queue);
    block_weights.ffn_output_bias =
        upload_tensor({1024}, weights.get(prefix + "ff.net.2.bias"), m_queue);

    decoder.blocks.push_back(std::move(block_weights));
  }

  return decoder;
}

flib::ftensor
FunGTMesher::decode_triplane(const flib::ftensor &image_tokens,
                             const std::string &diagnostic_directory) {
  // Three learned 32x32 planes, each with 1024 feature channels.
  auto triplane_tokens = upload_tensor(
      {3, 1024, 32, 32}, m_weight_loader.get("tokenizer.embeddings"), m_queue);
  auto decoder_weights = load_decoder_weights();

  return m_triplane_decoder->decode(triplane_tokens, image_tokens,
                                    decoder_weights, diagnostic_directory,
                                    m_queue);
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

void FunGTMesher::image_encoder(flib::ftensor &tokens,
                                const std::string &diagnostic_directory) {
  for (std::size_t layer = 0; layer < 12; ++layer) {
    auto weights = m_weight_loader.scope("image_tokenizer");
    const std::string layer_prefix =
        "encoder.layer." + std::to_string(layer) + ".";

    fgtm_tools::EncoderLayerWeights device_weights;
    device_weights.norm_before_weight = upload_tensor(
        {768}, weights.get(layer_prefix + "layernorm_before.weight"), m_queue);
    device_weights.norm_before_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "layernorm_before.bias"), m_queue);

    device_weights.query_weight = upload_tensor(
        {768, 768},
        weights.get(layer_prefix + "attention.attention.query.weight"),
        m_queue);
    device_weights.query_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.attention.query.bias"),
        m_queue);
    device_weights.key_weight = upload_tensor(
        {768, 768},
        weights.get(layer_prefix + "attention.attention.key.weight"), m_queue);
    device_weights.key_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.attention.key.bias"),
        m_queue);
    device_weights.value_weight = upload_tensor(
        {768, 768},
        weights.get(layer_prefix + "attention.attention.value.weight"),
        m_queue);
    device_weights.value_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.attention.value.bias"),
        m_queue);

    device_weights.attention_weight = upload_tensor(
        {768, 768}, weights.get(layer_prefix + "attention.output.dense.weight"),
        m_queue);
    device_weights.attention_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "attention.output.dense.bias"),
        m_queue);

    device_weights.norm_after_weight = upload_tensor(
        {768}, weights.get(layer_prefix + "layernorm_after.weight"), m_queue);
    device_weights.norm_after_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "layernorm_after.bias"), m_queue);

    device_weights.ffn_expand_weight = upload_tensor(
        {3072, 768}, weights.get(layer_prefix + "intermediate.dense.weight"),
        m_queue);
    device_weights.ffn_expand_bias = upload_tensor(
        {3072}, weights.get(layer_prefix + "intermediate.dense.bias"), m_queue);
    device_weights.ffn_reduce_weight = upload_tensor(
        {768, 3072}, weights.get(layer_prefix + "output.dense.weight"),
        m_queue);
    device_weights.ffn_reduce_bias = upload_tensor(
        {768}, weights.get(layer_prefix + "output.dense.bias"), m_queue);

    m_image_encoder->set_weights(std::move(device_weights));
    m_image_encoder->encode_layer(tokens, m_queue);

    export_image_features(
        tokens, (std::filesystem::path(diagnostic_directory) /
                 ("funlib_after_layer" + std::to_string(layer) + ".bin"))
                    .string());
  }

  auto final_norm_weight = upload_tensor(
      {768}, m_weight_loader.get("image_tokenizer.model.layernorm.weight"),
      m_queue);
  auto final_norm_bias = upload_tensor(
      {768}, m_weight_loader.get("image_tokenizer.model.layernorm.bias"),
      m_queue);

  tokens = flib::operations::layer_norm(tokens, final_norm_weight,
                                        final_norm_bias, 1.0e-12f, m_queue);
}
