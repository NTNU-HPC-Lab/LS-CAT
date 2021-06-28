#include "includes.h"
__global__ void Bprop1(const float* in, float* dsyn1, const float* dlayer1, const float alpha)
{
int i = blockDim.y*blockIdx.y + threadIdx.y; //28*28
int j = threadIdx.x;                         //256
int k = blockIdx.x;                          //Data.count

atomicAdd(&dsyn1[i*256 + j], dlayer1[k*256 + j] * in[k*28*28 + i] * alpha);
}