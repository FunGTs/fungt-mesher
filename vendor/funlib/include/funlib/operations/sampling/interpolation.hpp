#if !defined(_INTERPOLATION_HPP_)
#define _INTERPOLATION_HPP_

#include <funlib/Tensor/tensor.hpp>

namespace flib::operations {

template <typename T>
Tensor<T> interpolate_bicubic(const Tensor<T> &input,
                              std::size_t output_height,
                              std::size_t output_width, sycl::queue Q,
                              sycl::event *kernel_event = nullptr);

} // namespace flib::operations

#endif // _INTERPOLATION_HPP_
