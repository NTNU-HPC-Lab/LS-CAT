#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void elementwise_mul_kernel(const float *data_a, const float *data_b, float *output, int n) {

int index = blockDim.x * blockIdx.x + threadIdx.x;
if (index < n) {
output[index] = data_a[index] * data_b[index];
}
}