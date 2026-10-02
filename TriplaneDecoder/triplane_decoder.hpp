#if !defined(_TRIPLANE_DECODER_HPP_)
#define _TRIPLANE_DECODER_HPP_

#include "decoder_block_weights.hpp"
#include <string>
#include <vector>

namespace fgtm_tools{

    // Decoder has 3 parts:
    // Self attention
    // Cross attention
    // GEGLU (Gated Linear Unit with GELU ) activation feed forward
    class TriplaneDecoder {

        private:
            static constexpr std::size_t m_token_count = 3072;
            static constexpr std::size_t m_embedding_size = 1024;
            static constexpr std::size_t m_head_count = 16;
            static constexpr std::size_t m_head_size = 64;

            flib::ftensor linear(const flib::ftensor &input,
                               const flib::ftensor &weight,
                               const flib::ftensor &bias,
                               sycl::queue &queue) const;

            flib::ftensor linear_without_bias(const flib::ftensor &input,
                                            const flib::ftensor &weight,
                                            sycl::queue &queue) const;

            void self_attention(flib::ftensor &triplane,
                              const BlockWeights &weights,
                              sycl::queue &queue) const;

            void cross_attention(flib::ftensor &triplane,
                               const flib::ftensor &image_tokens,
                               const BlockWeights &weights,
                               sycl::queue &queue) const;

            void feed_forward(flib::ftensor &triplane,
                            const BlockWeights &weights,
                            sycl::queue &queue) const;

            void decode_block(flib::ftensor &triplane,
                            const flib::ftensor &image_tokens,
                            const BlockWeights &weights,
                            sycl::queue &queue) const;

            void save_tensor(const flib::ftensor &tensor,
                             const std::string &path,
                             sycl::queue &queue) const;
            
            

        public:
          TriplaneDecoder() = default;
          ~TriplaneDecoder() = default;

          flib::ftensor decode(const flib::ftensor& triplane_tokens,
                               const flib::ftensor& image_tokens,
                               const DecoderWeights& weights,
                               const std::string &diagnostic_directory,
                               sycl::queue &queue) const;
    };





}

#endif // _TRIPLANE_DECODER_HPP_
