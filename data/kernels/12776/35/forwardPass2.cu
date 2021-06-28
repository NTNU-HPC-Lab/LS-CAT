#include "includes.h"
__global__ void forwardPass2(float* layer1, float* syn2, float* out)
{
int l = blockDim.x*blockIdx.x + threadIdx.x;
int Y = 128;
int Z = 10;

#pragma unroll
for (int j=0; j < Y; ++j)
out[l] += layer1[j] * syn2[j*Z + l];

out[l] = 1.0/(1.0 + exp(out[l]));
}