#include "nerf.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

flib::ftensor fgtm_tools::NeRF::linear(const flib::ftensor &input,
                                       const NeRFData &data,
                                       sycl::queue &queue) {
    auto output = flib::tensor_operations::gemm_batched(
        input, data.weight.transpose(), queue);
    return flib::operations::add_bias(output, data.bias, queue);
}

flib::ftensor fgtm_tools::NeRF::generate_grid(std::size_t point_offset,
                                             std::size_t point_count,
                                             sycl::queue &queue) {
    flib::ftensor pos({point_count, 3}, queue);

    float step = (2.0f * m_radius) / (m_grid_size - 1);
    float radius = m_radius;
    std::size_t grid_size = m_grid_size;
    sycl::range<1> global_size(point_count);
    float *position_data = pos.device_data();
    sycl::event event  = queue.submit([=](sycl::handler &h){

        h.parallel_for(global_size,[=](sycl::item<1> item){

            std::size_t batch_point = item.get_id(0);
            std::size_t point = point_offset + batch_point;

            std::size_t z = point / (grid_size * grid_size);
            std::size_t remaining = point % (grid_size * grid_size);
            std::size_t y = remaining / grid_size;
            std::size_t x = remaining % grid_size;

            // Convert world-space coordinates in [-radius, radius] to the
            // normalized [-1, 1] coordinates expected by triplane_sample.
            position_data[batch_point * 3] = (-radius + static_cast<float>(x) * step) / radius;

            position_data[batch_point * 3 + 1] = (-radius + static_cast<float>(y) * step) / radius;

            position_data[batch_point * 3 + 2] = (-radius + static_cast<float>(z) * step) / radius;


        });


    });
    event.wait();


    return pos; 

}

flib::ftensor fgtm_tools::NeRF::forward_propagation(flib::ftensor features,
                                                    const NeRFWeights &weights,
                                                    sycl::queue &queue) {

  if (weights.layers.size() != 10) {
    throw std::invalid_argument("NeRF expects exactly 10 linear layers");
  }

  for (std::size_t i = 0; i + 1 < weights.layers.size(); ++i) {
    features = linear(features, weights.layers[i], queue);
    features = flib::operations::silu(features, queue);
  }

  // Final layer: Linear only.
  return linear(features, weights.layers.back(), queue);
}

flib::ftensor fgtm_tools::NeRF::compute_density(const flib::ftensor &triplane,const NeRFWeights &weights, sycl::queue &queue) {
    flib::ftensor density;

    // NOTE: A 128^3 grid exceeds CUDA's 65,535 work-group limit in Funlib's
    // tiled GEMM, so sampling and MLP inference run in smaller point batches.
    for (std::size_t offset = 0; offset < m_num_points;
         offset += m_points_per_batch) {
      const std::size_t point_count =
          std::min(m_points_per_batch, m_num_points - offset);
      auto grid_pos = generate_grid(offset, point_count, queue);

      // Three planes x 40 features = 120 features.
      auto features =
          flib::operations::triplane_sample(triplane, grid_pos, queue);
      auto output = forward_propagation(std::move(features), weights, queue);
      auto batch_density =
          flib::operations::shifted_exp_column(output, 0, -1.0f, queue);

      if (offset == 0) {
        density = std::move(batch_density);
      } else {
        density = flib::tensor_operations::concatenate(
            density, batch_density, 0, queue);
      }
    }

    density.reshape({m_grid_size, m_grid_size, m_grid_size});

    return density;

}
