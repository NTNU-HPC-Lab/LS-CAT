#include "includes.h"
__global__ void Bprop2(const float* out, const float* layer1, float* dsyn2, const int count, const float alpha)
{
int i = blockDim.y*blockIdx.y + threadIdx.y; //256
int j = blockDim.x*blockIdx.x + threadIdx.x; //4
//int k = blockIdx.x;  //Data.count

atomicAdd(&dsyn2[i*4 + j], out[j] * layer1[256*(count) + i] * alpha);
}