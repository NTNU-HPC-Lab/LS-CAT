#pragma once

#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void pow_grad_cpu(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out);

template <typename T>
void pow_grad(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
template <typename T>
void pow_grad_device(Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out);

template <typename T>
void pow_grad_device(cudaStream_t custream, Tensor<T> *x, int power, Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
