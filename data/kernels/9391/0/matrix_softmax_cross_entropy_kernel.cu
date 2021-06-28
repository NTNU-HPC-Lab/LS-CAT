#include "includes.h"

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)




__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol, const float *input_a, const float *input_b, float *output) {
// Dynamic shared memory, size provided at kernel launch.
extern __shared__ float loss_per_row[];
// Two dimensional thread blocks.
int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x
+ threadIdx.x;
if (y >= nrow) {
return;
}
input_a += y * ncol;
input_b += y * ncol;
float maxval = *input_a;
// Find max for a row.
for (int x = 1; x < ncol; ++x) {
maxval = max(maxval, input_a[x]);
}
// Deduct by max for a row, and raise to exp.
float sum = 0;
for (int x = 0; x < ncol; ++x) {
sum += exp(input_a[x] - maxval);
}
// Compute per-row loss.
float loss = 0;
for (int x = 0; x < ncol; ++x) {
loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
}
loss_per_row[y] = loss;
__syncthreads();
// Compute reduce_mean across rows.
float mean_loss = 0;
// Use a single thread to reduce mean across rows.
if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
for (int i = 0; i < nrow; ++i) {
mean_loss += loss_per_row[i];
}
mean_loss /= nrow;
output[0] = mean_loss;
}
}