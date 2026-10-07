#if !defined(_TRIPLANE_SAMPLING_HPP_)
#define _TRIPLANE_SAMPLING_HPP_

#include <funlib/Tensor/tensor.hpp>

namespace flib::operations {

template <typename T>
Tensor<T> triplane_sample(const Tensor<T> &triplane,
                          const Tensor<T> &positions, sycl::queue Q,
                          sycl::event *kernel_event = nullptr);

} // namespace flib::operations

#endif // _TRIPLANE_SAMPLING_HPP_
