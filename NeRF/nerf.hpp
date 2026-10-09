#if !defined(_NERF_HPP_)
#define _NERF_HPP_
//Neural Radiance Field class
#include<funlib/funlib.hpp>
#include<vector>
namespace fgtm_tools {

    struct NeRFData{

        flib::ftensor weight; 
        flib::ftensor bias; 


    };
    struct NeRFWeights{
        std::vector<NeRFData> layers;
    };


    class NeRF {

        private: 
            int m_grid_size  = 64; 
            int m_num_points = m_grid_size * m_grid_size * m_grid_size;
            float m_radius = 0.87f;

            // Applies one linear layer:
            // [N,D] x [O,D]Transpose + [O] -> [N, O ]
            flib::ftensor linear(const flib::ftensor &input,
                                     const NeRFData &data, sycl::queue &queue);

            // Generates the world grid in [-0.87,0.87] and normalizes its
            // [N,3] positions to [-1,1] for triplane sampling.
            flib::ftensor generate_grid(sycl::queue &queue);

            flib::ftensor forward_propagation(flib::ftensor features,
                                              const NeRFWeights &weights,
                                              sycl::queue &queue);

          public:
            NeRF() = default;
            ~NeRF() = default;
            flib::ftensor compute_density(const flib::ftensor &triplane, const NeRFWeights &weights,sycl::queue &queue); 
    };
}


#endif // _NERF_HPP_
