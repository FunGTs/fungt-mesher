#if !defined(_TRANSPOSED_TENSOR_VIEW_HPP_)
#define _TRANSPOSED_TENSOR_VIEW_HPP_

namespace flib {

template <typename T> class Tensor;

namespace detail {

//Wrap a tensor
template<typename T>
class TransposedTensorView{

    private:
        const flib::Tensor<T> *m_tensor; 

    friend class flib::Tensor<T>;
    explicit TransposedTensorView(const flib::Tensor<T> &tensor){
        m_tensor = &tensor;
    }

    public:
    
    const flib::Tensor<T>& tensor() const {
        return *m_tensor;
    }


};


template<typename T>
struct tensor_operand_traits;


//Traits specialization:

//Raw tensor
template<typename T> 
struct tensor_operand_traits<flib::Tensor<T>>{

    using value_type = T; 
    static const flib::Tensor<T> & tensor(const flib::Tensor<T>& value){
        return value; 
    }
    static constexpr bool transposed = false;

};

//Transposed Tensor
template<typename T>
struct tensor_operand_traits <TransposedTensorView<T>>{
    
    using value_type = T; 
    static const flib::Tensor<T> & tensor(const TransposedTensorView<T>& value ){
        return value.tensor(); 
    }
    static constexpr bool transposed = true;


};


} // namespace detail

} // namespace flib

#endif // _TRANSPOSED_TENSOR_VIEW_HPP_
