#if !defined(_TENSOR_OPERATIONS_HPP_)
#define _TENSOR_OPERATIONS_HPP_
#include <funlib/Tensor/tensor.hpp>
#include <type_traits>

namespace flib {

class tensor_operations {

  // This class is used to perform operations on the Tensor class
  // It is a friend class of the Tensor class
  // It is used to perform operations on the Tensor class
  // It is a friend class of the sycl_handler class
  // It is used to perform operations on the Tensor class
public:
  template <typename T>
  static Tensor<T> gemm(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q,
                        sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> gemm_naive(const Tensor<T> &A, const Tensor<T> &B,
                              sycl::queue Q,
                              sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> matxvec(const Tensor<T> &A, const Tensor<T> &B,
                           sycl::queue Q, sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> gemmTiled(const Tensor<T> &A, const Tensor<T> &B,
                             sycl::queue Q,
                             sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> gemm_blocked2x2(const Tensor<T> &A, const Tensor<T> &B,
                                   sycl::queue Q,
                                   sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> gemm_tiled_blocked2x2(const Tensor<T> &A, const Tensor<T> &B,
                                         sycl::queue Q,
                                         sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> permute(const Tensor<T> &input,
                           const std::vector<std::size_t> &order, sycl::queue Q,
                           sycl::event *kernel_event = nullptr);
  template <typename T>
  static Tensor<T> concatenate(const Tensor<T> &left, const Tensor<T> &right,
                               std::size_t axis, sycl::queue Q,
                               sycl::event *kernel_event = nullptr);

  // dot product of two vectors or two one dimensional sets
  template <typename T>
  static T dot(const Tensor<T> &A, const Tensor<T> &B, sycl::queue Q);
  template <typename T> static T reduction(const Tensor<T> &A, sycl::queue Q);

  //using traits: 
  template <typename Left, typename Right>
  static Tensor<typename detail::tensor_operand_traits<Left>::value_type>
  gemm_batched(const Left &left, const Right &right, sycl::queue queue,
               sycl::event *kernel_event = nullptr) {
    using LeftTraits = detail::tensor_operand_traits<Left>;
    using RightTraits = detail::tensor_operand_traits<Right>;
    using T = typename LeftTraits::value_type;

    static_assert(std::is_same_v<T, typename RightTraits::value_type>,
                  "GEMM tensors must use the same data type");

    return gemm_batched_impl(LeftTraits::tensor(left),
                             RightTraits::tensor(right), queue,
                             LeftTraits::transposed, RightTraits::transposed,
                             kernel_event);
  }

private:
  template <typename T>
  static Tensor<T> gemm_batched_impl(const Tensor<T> &A, const Tensor<T> &B,
                                     sycl::queue Q, bool transpose_A,
                                     bool transpose_B,
                                     sycl::event *kernel_event);

};
// class sycl_handler;
// Matrix times a vector Ax = b or Matrix times a matrix AB = C

}; // namespace flib

#endif // _TENSOR_OPERATIONS_HPP_
