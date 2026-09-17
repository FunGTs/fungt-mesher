#if !defined(_NORMALIZATION_HPP_)
#define _NORMALIZATION_HPP_

#include <funlib/Tensor/tensor.hpp>

namespace flib::operations {

template <typename T>
Tensor<T> layer_norm(const Tensor<T> &input, const Tensor<T> &gamma,
                     const Tensor<T> &beta, T epsilon, sycl::queue Q,
                     sycl::event *kernel_event = nullptr);

} // namespace flib::operations

#endif // _NORMALIZATION_HPP_
