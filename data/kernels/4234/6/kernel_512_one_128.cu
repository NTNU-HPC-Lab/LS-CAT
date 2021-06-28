#include "includes.h"
__global__ void kernel_512_one_128(float *A, float *B, float *bnBias, float *bnScale, float *C) {
int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
int ind = line*128 + in_channel;

extern __shared__ float shared_[];
float *weights = shared_ + 512*4, *output = weights + 128*64, *input = shared_;
float *bias = output + 4*128, *scale = bias + 128;

for (int i = 0; i < 4; i++)
input[ind + i*512] = A[tile*2048 + i*512 + ind];
bias[in_channel] = bnBias[in_channel];
scale[in_channel] = bnScale[in_channel];
output[ind] = 0.0f;
__syncthreads();

for (int k = 0; k < 512; k += 64) {
float *B_start = B + k*128;
for (int i = 0; i < 16; i++)
weights[ind + i*512] = B_start[i*512 + ind];
__syncthreads();

float *A_start = input + k;
for (int p = 0; p < 64; p++) {
output[ind] += A_start[line*512 + p] * weights[in_channel + p*128];
}
__syncthreads();
}

float *C_start = C + tile*512, res = scale[in_channel] * output[ind] + bias[in_channel];
C_start[ind] = res > 0 ? res : 0;
}