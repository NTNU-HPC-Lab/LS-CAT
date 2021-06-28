#include "includes.h"
__global__ void kSqSumColumnwise(float* mat, float* target, unsigned int width, unsigned int height) {
__shared__ float sum_vals[32];
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

target[blockIdx.x] = cur_sum;
}
}