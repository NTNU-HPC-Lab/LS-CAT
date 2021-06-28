#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void marix_multiply_by_const(const float *d_input, float *d_output, float val, int n) {
int index = blockDim.x * blockIdx.x + threadIdx.x;
if (index < n) {
d_output[index] = d_input[index] * val;
}
}