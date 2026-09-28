#include "image_encoder.hpp"

#include <utility>

namespace fgtm_tools {

void ImageEncoder::set_weights(EncoderLayerWeights &&weights) {
  m_weights = std::move(weights);
}

flib::ftensor ImageEncoder::linear(const flib::ftensor &input,
                                   const flib::ftensor &weight,
                                   const flib::ftensor &bias,
                                   sycl::queue &queue) const {
  auto projected =
      flib::tensor_operations::gemm_batched(input, weight.transpose(), queue);
  return flib::operations::add_bias(projected, bias, queue);
}

void ImageEncoder::encode_layer(flib::ftensor &image_tokens,
                                sycl::queue &queue) const {
  constexpr std::size_t token_count = 197;
  constexpr std::size_t embedding_size = 768;
  constexpr std::size_t head_count = 12;
  constexpr std::size_t head_size = 64;
  constexpr float epsilon = 1.0e-12f;

  auto normalized =
      flib::operations::layer_norm(image_tokens, m_weights.norm_before_weight,
                                   m_weights.norm_before_bias, epsilon, queue);

  auto query =
      linear(normalized, m_weights.query_weight, m_weights.query_bias, queue);
  auto key =
      linear(normalized, m_weights.key_weight, m_weights.key_bias, queue);
  auto value =
      linear(normalized, m_weights.value_weight, m_weights.value_bias, queue);

  query.reshape({1, token_count, head_count, head_size});
  key.reshape({1, token_count, head_count, head_size});
  value.reshape({1, token_count, head_count, head_size});

  auto attention = flib::operations::scaled_dot_product_attention(
      query, key, value, head_count, queue);

  attention.reshape({token_count, embedding_size});

  auto attention_output = linear(attention, m_weights.attention_weight,
                                 m_weights.attention_bias, queue);
  auto first_residual =
      flib::operations::add(image_tokens, attention_output, queue);

  auto normalized_residual =
      flib::operations::layer_norm(first_residual, m_weights.norm_after_weight,
                                   m_weights.norm_after_bias, epsilon, queue);

  auto hidden = linear(normalized_residual, m_weights.ffn_expand_weight,
                       m_weights.ffn_expand_bias, queue);
  hidden = flib::operations::gelu(hidden, queue);

  auto feed_forward = linear(hidden, m_weights.ffn_reduce_weight,
                             m_weights.ffn_reduce_bias, queue);
  image_tokens = flib::operations::add(first_residual, feed_forward, queue);
}

} // namespace fgtm_tools
