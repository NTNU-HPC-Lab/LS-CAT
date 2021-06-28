#include "includes.h"
__global__ void Bprop2(const float* layer1, float* dsyn2, const float* out, const float alpha)
{
int i = threadIdx.x; //256
int j = blockDim.y*blockIdx.y + threadIdx.y; //10
int k = blockIdx.x;  //Data.count

atomicAdd(&dsyn2[i*10 + j], out[k*10 + j] * layer1[256*k + i] * alpha);
}