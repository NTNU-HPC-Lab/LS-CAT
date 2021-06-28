#include "includes.h"
__global__ void kNormLimitColumnwise(float* mat, float* target, float norm, unsigned int width, unsigned int height) {
__shared__ float sum_vals[33];
float cur_sum = 0;
for (unsigned int i = threadIdx.x; i < height; i += 32) {
cur_sum += mat[blockIdx.x * height + i] * mat[blockIdx.x * height + i];
}
sum_vals[threadIdx.x] = cur_sum;
__syncthreads();
if (threadIdx.x == 0) {
cur_sum = 0;
for (unsigned int i = 0; i < 32; i++)
cur_sum += sum_vals[i];
sum_vals[32] = norm > cur_sum ? 1 : norm / sqrt(cur_sum);
}
float scale = sum_vals[32];
for (unsigned int i = threadIdx.x; i < height; i += 32) {
target[blockIdx.x * height + i] = mat[blockIdx.x * height + i] * scale;
}
__syncthreads();
}