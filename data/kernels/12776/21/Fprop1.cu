#include "includes.h"
__global__ void Fprop1(const float* in, const float* syn1, float* layer1)
{
int i = threadIdx.x;                         //256
int j = blockDim.y*blockIdx.y + threadIdx.y; //64
int k = blockIdx.x;                          //Data.count
atomicAdd(&layer1[256*k + i], in[64*k + j] * syn1[j*256 + i]);
}