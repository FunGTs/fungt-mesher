#if !defined(_SOFTMAX_HPP_)
#define _SOFTMAX_HPP_
#include <funlib/Tensor/tensor.hpp>
namespace flib::operations {

template <typename T>
Tensor<T> softmax(const Tensor<T> &input, sycl::queue Q,
                  sycl::event *kernel_event = nullptr);

template <typename T>
Tensor<T> scaled_softmax(const Tensor<T> &input, T scale, sycl::queue Q,
                         sycl::event *kernel_event = nullptr);

} // namespace flib::operations

#endif // _SOFTMAX_HPP_
