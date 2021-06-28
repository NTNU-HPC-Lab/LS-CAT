#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void relu_kernel(const float *input, float *output, int n) {
int index = blockDim.x * blockIdx.x + threadIdx.x;
if (index < n) {
float element = input[index];
if (element <= 0) {
output[index] = 0;
} else {
output[index] = element;
}
}
}