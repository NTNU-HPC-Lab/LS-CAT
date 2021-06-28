#include "includes.h"
__global__ void forwardPass1(float* in, float* syn1, float* layer1)
{
int l = blockDim.x*blockIdx.x + threadIdx.x;
int j = blockDim.y*blockIdx.y + threadIdx.y;
int Y = 128;

atomicAdd(&layer1[l] , in[j] * syn1[j*Y + l]);

layer1[l] = 1.0/(1.0 + exp(layer1[l]));
}