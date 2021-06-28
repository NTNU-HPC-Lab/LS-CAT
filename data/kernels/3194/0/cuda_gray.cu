#include "includes.h"

__global__ void cuda_gray(unsigned char *input, int offset, int streamSize, unsigned char* gray, int size) {

int gray_idx = (offset/3) + (blockIdx.x * blockDim.x + threadIdx.x);
int rgb_idx = (offset) + ((blockIdx.x * blockDim.x + threadIdx.x) * 3);

if (((blockIdx.x * blockDim.x + threadIdx.x)*3)>=streamSize || gray_idx>=size) {
return;
}

gray[gray_idx] = (gray_value[0] * input[rgb_idx]) + (gray_value[1] * input[rgb_idx + 1]) + (gray_value[2] * input[rgb_idx + 2]);
}