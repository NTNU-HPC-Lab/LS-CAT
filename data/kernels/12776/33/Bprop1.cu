#include "includes.h"
__global__ void Bprop1(const float* dlayer1, const float* dlayer1i, const float* dlayer1o, const float* in, float* dsyn1, float* dsyn1i, float* dsyn1o, const float alpha)
{
int i = blockDim.y*blockIdx.y + threadIdx.y; //64
int j = threadIdx.x;                         //256
int k = blockIdx.x;                          //Data.count

atomicAdd(&dsyn1[i*256 + j],  dlayer1[k*256 + j]  * in[k*64 + i] * alpha);
atomicAdd(&dsyn1i[i*256 + j], dlayer1i[k*256 + j] * in[k*64 + i] * alpha);
atomicAdd(&dsyn1o[i*256 + j], dlayer1o[k*256 + j] * in[k*64 + i] * alpha);
}