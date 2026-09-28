#if !defined(_IMAGE_ENCODER_HPP_)
#define _IMAGE_ENCODER_HPP_

#include <funlib/funlib.hpp>

namespace fgtm_tools {

struct EncoderLayerWeights {
  flib::ftensor norm_before_weight;
  flib::ftensor norm_before_bias;

  flib::ftensor query_weight;
  flib::ftensor query_bias;
  flib::ftensor key_weight;
  flib::ftensor key_bias;
  flib::ftensor value_weight;
  flib::ftensor value_bias;

  flib::ftensor attention_weight;
  flib::ftensor attention_bias;

  flib::ftensor norm_after_weight;
  flib::ftensor norm_after_bias;

  flib::ftensor ffn_expand_weight;
  flib::ftensor ffn_expand_bias;
  flib::ftensor ffn_reduce_weight;
  flib::ftensor ffn_reduce_bias;
};

class ImageEncoder {
private:
  EncoderLayerWeights m_weights;

  flib::ftensor linear(const flib::ftensor &input, const flib::ftensor &weight,
                       const flib::ftensor &bias, sycl::queue &queue) const;

public:
  ImageEncoder() = default;
  ~ImageEncoder() = default;

  void set_weights(EncoderLayerWeights &&weights);
  void encode_layer(flib::ftensor &image_tokens, sycl::queue &queue) const;
};

} // namespace fgtm_tools

#endif // _IMAGE_ENCODER_HPP_
