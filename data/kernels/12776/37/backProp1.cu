#include "includes.h"
__global__ void backProp1(float* in, float* dsyn1, float* layer1, float* syn2, float* label, float* out)
{
int j = blockDim.x*blockIdx.x + threadIdx.x;
int k = blockDim.y*blockIdx.y + threadIdx.y;
float error = 0.0;

#pragma unroll
for (int l=0; l < 10; ++l)
error += (label[l] - out[l]) * syn2[k*10 + l];
float delta = error * (layer1[k]*(1-layer1[k]));
dsyn1[j*128 + k] += delta * in[j] / (60000.0/10.0);
}