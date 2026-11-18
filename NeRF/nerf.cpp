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

flib::ftensor fgtm_tools::NeRF::compute_color(const flib::ftensor &triplane,
                                              const flib::ftensor &positions,
                                              const NeRFWeights &weights,
                                              sycl::queue &queue) {

    if (positions.getRank() != 2 || positions.getShape().size() != 2 || positions.getShape()[1] != 3) {
    throw std::invalid_argument(
        "NeRF color query expects positions with shape [N, 3]");
    }

    if (!positions.is_device() || !positions.is_accessible_from(queue)) {
        throw std::invalid_argument(
            "NeRF color positions must be accessible from the queue");
    }

    //Get total of vertices:
    const std::size_t vertex_count = positions.getShape()[0];
    flib::ftensor colors({vertex_count, 3}, queue);
    if (vertex_count == 0) {
        return colors;
    }
    flib::ftensor norm_pos({vertex_count, 3}, queue);

    const float* pos_data_ptr = positions.device_data();

    float * norm_pos_ptr = norm_pos.device_data();
    float radius = m_radius;  

    queue.submit([=](sycl::handler &h){
        h.parallel_for(sycl::range<1>(vertex_count), [=](sycl::item<1> item){
            const std::size_t vertex = item.get_id(0);

            std::size_t x = vertex * 3;
            std::size_t y = vertex * 3 + 1;
            std::size_t z = vertex * 3 + 2;

            norm_pos_ptr[x] = pos_data_ptr[x] / radius;
            norm_pos_ptr[y] = pos_data_ptr[y] / radius;
            norm_pos_ptr[z] = pos_data_ptr[z] / radius;

        });
    }).wait();


    auto features = flib::operations::triplane_sample(
        triplane,
        norm_pos, queue);

    auto output = forward_propagation(std::move(features), weights, queue);

    // The decoder output is [density, red, green, blue].
    if (output.getRank() != 2 || output.getShape()[0] != vertex_count || output.getShape()[1] < 4) {
      throw std::invalid_argument( "NeRF decoder must produce at least four outputs");
    }

    const float *output_data = output.device_data();
    float *color_data = colors.device_data();

    queue.submit([=](sycl::handler &handler) {
          handler.parallel_for(
              sycl::range<1>(vertex_count), [=](sycl::item<1> item) {
                const std::size_t vertex = item.get_id(0);

                for (std::size_t channel = 0; channel < 3; ++channel) {
                  const float logit = output_data[vertex * 4 + channel + 1];

                  color_data[vertex * 3 + channel] =
                      1.0f / (1.0f + sycl::exp(-logit));
                }
              });
        }).wait();

    return colors;
}
