#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void reduced_sum_axis_zero(const float *input_data, float *output_data, int input_n, int output_n) {
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < output_n) {
output_data[idx] = 0.0;
for (int i = 0; i < input_n / output_n; i++) {
output_data[idx] += input_data[i * output_n + idx];
}
}
}