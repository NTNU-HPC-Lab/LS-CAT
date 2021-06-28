/**
 * @file tanh_internal.h
 * @author Daniel Nichols
 * @author Florent Lopez
 * @version 1.0
 * @date 2019-02-23
 *
 * @copyright Copyright (c) 2019
 */
#pragma once
#include <math.h>
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

/** Computes the element-wise tanh function.
 * @tparam T
 * @param x
 */
template <typename T>
void tanh_full(Tensor<T> *x, Tensor<T> *out);

#if defined(MAGMADNN_HAVE_CUDA)
/** Computes the tanh function element-wise on the tensor x
 * @tparam T
 * @param x
 */
template <typename T>
void tanh_full_device(Tensor<T> *x, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
