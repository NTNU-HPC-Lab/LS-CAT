#include "includes.h"
__global__ void kArgMaxRowwise(float* mat, float* target, unsigned int width, unsigned int height) {
__shared__ float max_vals[32];
__shared__ unsigned int max_args[32];
float cur_max = -2e38;
unsigned int cur_arg = 0;
float val = 0;

for (unsigned int i = threadIdx.x; i < width; i += 32) {
val = mat[blockIdx.x * width + i];

if (val > cur_max) {
cur_max = val;
cur_arg = i;
}
}

max_vals[threadIdx.x] = cur_max;
max_args[threadIdx.x] = cur_arg;

__syncthreads();

if (threadIdx.x == 0) {
cur_max = -2e38;
cur_arg = 0;

for (unsigned int i = 0; i < 32; i++)
if (max_vals[i] > cur_max) {
cur_max = max_vals[i];
cur_arg = max_args[i];
}

target[blockIdx.x] = cur_arg;
}
}