#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void matrix_elementwise_add(const float *a, const float *b, float *c, int n) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) {
c[index] = a[index] + b[index];
}
}