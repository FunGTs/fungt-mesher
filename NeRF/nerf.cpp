#include "nerf.hpp"

#include <stdexcept>
#include <utility>

flib::ftensor fgtm_tools::NeRF::linear(const flib::ftensor &input,
                                       const NeRFData &data,
                                       sycl::queue &queue) {
    auto output = flib::tensor_operations::gemm_batched(
        input, data.weight.transpose(), queue);
    return flib::operations::add_bias(output, data.bias, queue);
}

flib::ftensor fgtm_tools::NeRF::generate_grid(sycl::queue &queue) {
    std::size_t N = static_cast<std::size_t>(m_num_points);
    flib::ftensor pos({N, 3}, queue);

    float step = (2.0f * m_radius) / (m_grid_size - 1);
    float radius = m_radius;
    std::size_t grid_size = m_grid_size;
    sycl::range<1> global_size(m_num_points);
    float *position_data = pos.device_data();
    sycl::event event  = queue.submit([=](sycl::handler &h){

        h.parallel_for(global_size,[=](sycl::item<1> item){

            std::size_t point = item.get_id(0);

            std::size_t z = point / (grid_size * grid_size);
            std::size_t remaining = point % (grid_size * grid_size);
            std::size_t y = remaining / grid_size;
            std::size_t x = remaining % grid_size;

            // Convert world-space coordinates in [-radius, radius] to the
            // normalized [-1, 1] coordinates expected by triplane_sample.
            position_data[point * 3] = (-radius + static_cast<float>(x) * step) / radius;

            position_data[point * 3 + 1] = (-radius + static_cast<float>(y) * step) / radius;

            position_data[point * 3 + 2] = (-radius + static_cast<float>(z) * step) / radius;


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
    
    
    auto grid_pos = generate_grid(queue);

    // Three planes x 40 features = 120 features
    // [262144,3] -> [262144,120]

    auto features = flib::operations::triplane_sample(triplane, grid_pos, queue);

    // [262144,120] -> [262144,4]
    auto output = forward_propagation(std::move(features), weights, queue);

    // exp(output[:,0] - 1) -> [262144]
    auto density =
        flib::operations::shifted_exp_column(output, 0, -1.0f, queue);

    density.reshape({64, 64, 64});

    return density;

}
