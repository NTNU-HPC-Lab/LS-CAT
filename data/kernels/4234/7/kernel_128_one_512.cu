#include "includes.h"
__global__ void kernel_128_one_512(float *A, float *B, float *bnBias, float *bnScale, float *C) {
int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
int ind = line*128 + in_channel;

extern __shared__ float shared_[];
float *weights = shared_ + 128*4, *output = weights + 128*64, *input = shared_;
float *bias = output + 4*128, *scale = bias + 128;

input[ind] = A[tile * 512 + ind];
bias[in_channel] = bnBias[part*128 + in_channel];
scale[in_channel] = bnScale[part*128+ in_channel];
output[ind] = 0.0f;
__syncthreads();

for (int k = 0; k < 128; k += 64) {
for (int i = 0; i < 16; i++)
weights[ind + 512*i] = B[(k + i*4 + line)*512 + part*128 + in_channel];
__syncthreads();

float *A_start = input + k;
for (int p = 0; p < 64; p++) {
output[ind] += A_start[line*128 + p] * weights[in_channel + p*128];
}
__syncthreads();
}

float *C_start = C + tile*2048 + part*128;
float res = scale[in_channel] * output[ind] + bias[in_channel];
C_start[line * 512 + in_channel] = res;
}