#if !defined(_ELEMENT_WISE_OPERATIONS_HPP_)
#define _ELEMENT_WISE_OPERATIONS_HPP_
#include <funlib/Tensor/tensor.hpp>

namespace flib::operations {

template <typename T>
Tensor<T> scale(const Tensor<T> &input, T value, sycl::queue Q,
                sycl::event *kernel_event = nullptr);

template <typename T>
Tensor<T> add(const Tensor<T> &left, const Tensor<T> &right, sycl::queue Q,
              sycl::event *kernel_event = nullptr);

template <typename T>
Tensor<T> gelu(const Tensor<T> &input, sycl::queue Q,
               sycl::event *kernel_event = nullptr);

} // namespace flib::operations

#endif // _ELEMENT_WISE_OPERATIONS_HPP_
