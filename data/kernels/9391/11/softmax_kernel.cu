#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void softmax_kernel(int64_t nrow, int64_t ncol, const float *input_data, float *output_data) {

// two dimensional thread blocks.
int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
if (y >= nrow) {
return;
}
// y_th row of input data
input_data += y * ncol;
output_data += y * ncol;
// find max for a row.
float maxval = *input_data;
for (int x = 1; x < ncol; ++x) {
maxval = max(maxval, input_data[x]);
}
// Deduct by max for a row, and raise to exp.
// in case of too large of exp, and the result will not be affected
float sum = 0;
for (int x = 0; x < ncol; ++x) {
sum += exp(input_data[x] - maxval);
}
// Compute per-row softmax.
for (int x = 0; x < ncol; ++x) {
output_data[x] = exp(input_data[x] - maxval) / sum;
}
}