#if !defined(_ATTENTION_OPERATIONS_HPP_)
#define _ATTENTION_OPERATIONS_HPP_

#include <funlib/Tensor/tensor.hpp>

namespace flib::operations {

template <typename T>
Tensor<T> split_heads(const Tensor<T> &input, sycl::queue Q,
                      sycl::event *kernel_event = nullptr);

template <typename T>
Tensor<T> join_heads(const Tensor<T> &input, sycl::queue Q,
                     sycl::event *kernel_event = nullptr);

template <typename T>
Tensor<T> scaled_dot_product_attention(const Tensor<T> &query,
                                       const Tensor<T> &key,
                                       const Tensor<T> &value,
                                       std::size_t head_count, sycl::queue Q);

} // namespace flib::operations

#endif // _ATTENTION_OPERATIONS_HPP_
