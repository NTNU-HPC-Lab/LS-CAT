#include "includes.h"
__global__ void kernel_1024_one_256(float *A, float *B, float *bnBias, float *bnScale, float *C) {
int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
int ind = line*256 + in_channel;

extern __shared__ float shared_[];
float *weights = shared_ + 1024*4, *output = weights + 256*16, *input = shared_;
float *bias = output + 4*256, *scale = bias + 256;

for (int i = 0; i < 4; i++)
input[ind + i*1024] = A[tile*4096 + i*1024 + ind];
bias[in_channel] = bnBias[in_channel];
scale[in_channel] = bnScale[in_channel];
output[ind] = 0.0f;
__syncthreads();

for (int k = 0; k < 1024; k += 16) {
float *B_start = B + k*256;
for (int i = 0; i < 4; i++)
weights[ind + i*1024] = B_start[i*1024 + ind];
__syncthreads();

float *A_start = input + k;
for (int p = 0; p < 16; p++) {
output[ind] += A_start[line*1024 + p] * weights[in_channel + p*256];
}
__syncthreads();
}

float *C_start = C + tile*1024, res = scale[in_channel] * output[ind] + bias[in_channel];
C_start[ind] = res > 0 ? res : 0;
}