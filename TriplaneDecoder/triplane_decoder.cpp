#include "triplane_decoder.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fgtm_tools{

    // Saves one decoder stage as raw float32 values for reference validation.
    void TriplaneDecoder::save_tensor(
        const flib::ftensor &tensor,
        const std::string &path,
        sycl::queue &queue) const {

        const auto values = tensor.to_host(queue);
        std::ofstream output(path, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to open decoder output: " + path);
        }

        output.write(
            reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!output) {
            throw std::runtime_error("Failed to write decoder output: " + path);
        }
    }

    // Projects every token from D to O features and applies one bias per
    // output feature. B is batch size, N is token count, D is input feature
    // size, and O is output feature size. Input is [B, N, D], while the
    // weight is stored as [O, D].
    flib::ftensor TriplaneDecoder::linear(
        const flib::ftensor &input,
        const flib::ftensor &weight,
        const flib::ftensor &bias,
        sycl::queue &queue) const {

        auto output = linear_without_bias(input, weight, queue);
        return flib::operations::add_bias(output, bias, queue);
    }

    // Projects [B, N, D] to [B, N, O] without bias. B is batch size, N is
    // token count, D is input feature size, and O is output feature size. The
    // input is temporarily flattened because funlib batched GEMM requires both
    // tensors to have the same rank, while decoder weights are rank-2 [O, D].
    flib::ftensor TriplaneDecoder::linear_without_bias(
        const flib::ftensor &input,
        const flib::ftensor &weight,
        sycl::queue &queue) const {

        if (input.getRank() != 3 || weight.getRank() != 2) {
            throw std::invalid_argument(
                "TriplaneDecoder linear expects input [B,N,D] and weight [O,D]");
        }

        const auto &input_shape = input.getShape();
        const auto &weight_shape = weight.getShape();

        if (input_shape[2] != weight_shape[1]) {
            throw std::invalid_argument(
                "TriplaneDecoder linear input and weight dimensions do not match");
        }

        // Flatten tokens for rank-2 GEMM: [B, N, D] -> [B*N, D]
        auto flattened = input;
        flattened.reshape({input_shape[0] * input_shape[1], input_shape[2]});

        auto output = flib::tensor_operations::gemm_batched(
            flattened, weight.transpose(), queue);

        // Restore the decoder layout: [B*N, O] -> [B, N, O]
        output.reshape({input_shape[0], input_shape[1], weight_shape[0]});
        return output;
    }

    // Self-attention lets every triplane token read information from all other
    // triplane tokens. B is batch size, N is token count, D is embedding size,
    // H is head count, and Dh is the feature size of one head (D / H).
    void TriplaneDecoder::self_attention(
        flib::ftensor &triplane,
        const BlockWeights &weights,
        sycl::queue &queue) const {

        constexpr float epsilon = 1.0e-5f;
        const auto &shape = triplane.getShape();
        const std::size_t batch_size = shape[0];
        const std::size_t token_count = shape[1];

        // Preserve [B, N, D] for the residual connection.
        auto residual = triplane;

        // Normalize every token before attention. Shape stays [B, N, D].
        auto normalized = flib::operations::layer_norm(
            triplane, weights.norm1, weights.norm1_bias, epsilon, queue);

        // Q, K and V all come from the triplane. Shape: [B, N, D].
        auto query = linear_without_bias(normalized, weights.self_query, queue);
        auto key = linear_without_bias(normalized, weights.self_key, queue);
        auto value = linear_without_bias(normalized, weights.self_value, queue);

        // Split D into H attention heads: [B, N, D] -> [B, N, H, Dh].
        query.reshape({batch_size, token_count, m_head_count, m_head_size});
        key.reshape({batch_size, token_count, m_head_count, m_head_size});
        value.reshape({batch_size, token_count, m_head_count, m_head_size});

        // Attention joins the heads and returns [B, N, D].
        auto attention = flib::operations::scaled_dot_product_attention(
            query, key, value, m_head_count, queue);

        // Mix the attention result and add the original triplane tokens.
        auto output = linear(attention, weights.self_output,
                             weights.self_output_bias, queue);
        triplane = flib::operations::add(residual, output, queue);
    }

    // Cross-attention transfers image information into the triplane. B is
    // batch size, Nt is triplane token count, Ni is image token count, D is
    // decoder embedding size, H is head count, and Dh is D / H.
    void TriplaneDecoder::cross_attention(
        flib::ftensor &triplane,
        const flib::ftensor &image_tokens,
        const BlockWeights &weights,
        sycl::queue &queue) const {

        constexpr float epsilon = 1.0e-5f;
        const auto &triplane_shape = triplane.getShape();
        const auto &image_shape = image_tokens.getShape();
        const std::size_t batch_size = triplane_shape[0];
        const std::size_t triplane_token_count = triplane_shape[1];
        const std::size_t image_token_count = image_shape[1];

        // Preserve [B, Nt, D] for the residual connection.
        auto residual = triplane;

        // Normalize triplane queries. Shape stays [B, Nt, D].
        auto normalized = flib::operations::layer_norm(
            triplane, weights.norm2, weights.norm2_bias, epsilon, queue);

        // Q comes from triplane tokens; K and V come from image tokens.
        auto query = linear_without_bias(normalized, weights.cross_query, queue);
        auto key = linear_without_bias(image_tokens, weights.cross_key, queue);
        auto value = linear_without_bias(image_tokens, weights.cross_value, queue);

        // Split D into H heads. Q has Nt tokens; K and V have Ni tokens.
        query.reshape(
            {batch_size, triplane_token_count, m_head_count, m_head_size});
        key.reshape(
            {batch_size, image_token_count, m_head_count, m_head_size});
        value.reshape(
            {batch_size, image_token_count, m_head_count, m_head_size});

        // Each triplane query reads from every encoded image token.
        auto attention = flib::operations::scaled_dot_product_attention(
            query, key, value, m_head_count, queue);

        // Return to [B, Nt, D] and add the triplane residual.
        auto output = linear(attention, weights.cross_output,
                             weights.cross_output_bias, queue);
        triplane = flib::operations::add(residual, output, queue);
    }

    // The feed-forward network processes each triplane token independently.
    // B is batch size, N is token count, D is embedding size, and F is the
    // GEGLU hidden size. Shapes: [B,N,D] -> [B,N,2F] -> [B,N,F] -> [B,N,D].
    void TriplaneDecoder::feed_forward(
        flib::ftensor &triplane,
        const BlockWeights &weights,
        sycl::queue &queue) const {

        constexpr float epsilon = 1.0e-5f;

        // Preserve [B, N, D] for the residual connection.
        auto residual = triplane;

        // Normalize every token before the feed-forward network.
        auto normalized = flib::operations::layer_norm(
            triplane, weights.norm3, weights.norm3_bias, epsilon, queue);

        // Expand D=1024 to 2F=8192 for the GEGLU value and gate halves.
        auto hidden = linear(normalized, weights.geglu,
                             weights.geglu_bias, queue);

        // Split 8192 into two 4096 halves and compute value * GELU(gate).
        hidden = flib::operations::geglu(hidden, queue);

        // Reduce F=4096 back to D=1024 and add the residual.
        auto output = linear(hidden, weights.ffn_output,
                             weights.ffn_output_bias, queue);
        triplane = flib::operations::add(residual, output, queue);
    }

    // Runs one decoder transformer block. B is batch size, N is the number of
    // triplane tokens, and D is the embedding size. Every stage preserves the
    // triplane shape [B, N, D] and applies its own residual connection.
    void TriplaneDecoder::decode_block(
        flib::ftensor &triplane,
        const flib::ftensor &image_tokens,
        const BlockWeights &weights,
        sycl::queue &queue) const {

        // Share information between the N triplane tokens.
        self_attention(triplane, weights, queue);

        // Transfer encoded image information into the triplane tokens.
        cross_attention(triplane, image_tokens, weights, queue);

        // Process each token with the GEGLU feed-forward network.
        feed_forward(triplane, weights, queue);
    }

    flib::ftensor TriplaneDecoder::decode(const flib::ftensor& triplane,
        const flib::ftensor &image_tokens,
        const DecoderWeights &weights,
        const std::string &diagnostic_directory,
        sycl::queue &queue) const {

        // Current triplane tensor shape
        // [planes, channels, height, width]
        // [3, 1024, 32, 32]

        // The Colab reference uses a direct contiguous reshape.
        // [3, 1024, 32, 32] -> [1, 1024, 3072]
        auto decoded = triplane;
        decoded.reshape({1, m_embedding_size, m_token_count});

        // Keep the learned triplane for the outer residual.
        auto residual = decoded;

        // Normalize the channels before the transformer blocks.
        decoded = flib::operations::group_norm(
            decoded, weights.norm, weights.norm_bias, 32, 1.0e-6f, queue);
        /*save_tensor(
            decoded,
            (std::filesystem::path(diagnostic_directory) /
             "funlib_after_norm.bin").string(),
            queue);*/

        // Use token-first transformer layout: [1, 1024, 3072] -> [1, 3072, 1024]
        decoded = flib::tensor_operations::permute(
            decoded, {0, 2, 1}, queue);

        // Project features into the decoder.
        decoded = linear(decoded, weights.projection_in,
                         weights.projection_in_bias, queue);
        /*save_tensor(
            decoded,
            (std::filesystem::path(diagnostic_directory) /
             "funlib_after_proj_in.bin").string(),
            queue);*/

        // Refine the triplane and inject image information.
        for (std::size_t block = 0; block < weights.blocks.size(); ++block) {
            decode_block(decoded, image_tokens, weights.blocks[block], queue);
            /*save_tensor(
                decoded,
                (std::filesystem::path(diagnostic_directory) /
                 ("funlib_after_block" + std::to_string(block) + ".bin"))
                    .string(),
                queue);*/
        }

        // Project back to the triplane representation.
        decoded = linear(decoded, weights.projection_out,
                         weights.projection_out_bias, queue);
        /*save_tensor(
            decoded,
            (std::filesystem::path(diagnostic_directory) /
             "funlib_after_proj_out.bin").string(),
            queue);*/

        // Restore channel-first layout: [1, 3072, 1024] -> [1, 1024, 3072]
        decoded = flib::tensor_operations::permute(
            decoded, {0, 2, 1}, queue);

        // Add the original learned triplane features.
        auto output = flib::operations::add(decoded, residual, queue);
        /*save_tensor(
            output,
            (std::filesystem::path(diagnostic_directory) /
             "funlib_final_output.bin").string(),
            queue);*/
        return output;
    }
        
}
