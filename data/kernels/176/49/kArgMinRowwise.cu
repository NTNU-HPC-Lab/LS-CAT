#include "includes.h"
__global__ void kArgMinRowwise(float* mat, float* target, unsigned int width, unsigned int height) {
__shared__ float min_vals[32];
__shared__ unsigned int min_args[32];
float cur_min = 2e38;
unsigned int cur_arg = 0;
float val = 0;

for (unsigned int i = threadIdx.x; i < width; i += 32) {
val = mat[blockIdx.x * width + i];

if (val < cur_min) {
cur_min = val;
cur_arg = i;
}
}

min_vals[threadIdx.x] = cur_min;
min_args[threadIdx.x] = cur_arg;

__syncthreads();

if (threadIdx.x == 0) {
cur_min = 2e38;
cur_arg = 0;

for (unsigned int i = 0; i < 32; i++)
if (min_vals[i] < cur_min) {
cur_min = min_vals[i];
cur_arg = min_args[i];
}

target[blockIdx.x] = cur_arg;
}
}