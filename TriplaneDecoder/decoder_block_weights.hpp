#if !defined(_DECODER_BLOCK_WEIGHTS_HPP_H)
#define _DECODER_BLOCK_WEIGHTS_HPP_H

#include <funlib/funlib.hpp>

// Contains weights belonging to one specific transformer block
struct BlockWeights {

  // Self-attention

  flib::ftensor norm1; //Layer norm before self attention
  flib::ftensor norm1_bias;

  flib::ftensor self_query;
  flib::ftensor self_key;
  flib::ftensor self_value;
  flib::ftensor self_output;
  flib::ftensor self_output_bias;

  // Cross-attention
  flib::ftensor norm2;
  flib::ftensor norm2_bias;

  flib::ftensor cross_query;
  flib::ftensor cross_key;
  flib::ftensor cross_value;
  flib::ftensor cross_output;
  flib::ftensor cross_output_bias;

  // GEGLU
  flib::ftensor norm3;
  flib::ftensor norm3_bias;

  flib::ftensor geglu;
  flib::ftensor geglu_bias;
  flib::ftensor ffn_output;
  flib::ftensor ffn_output_bias;
};
// TriplaneDecoderWeights contains weights used once for the entire decoder:
struct DecoderWeights {
  flib::ftensor norm;
  flib::ftensor norm_bias;

  flib::ftensor projection_in;
  flib::ftensor projection_in_bias;

  std::vector<BlockWeights> blocks;

  flib::ftensor projection_out;
  flib::ftensor projection_out_bias;
};

#endif // _DECODER_BLOCK_WEIGHTS_HPP_H
