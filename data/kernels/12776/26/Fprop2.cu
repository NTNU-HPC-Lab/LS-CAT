#include "includes.h"
__global__ void Fprop2(const float* layer1, const float* syn2, float* out, const int offset)
{
int i = blockDim.x*blockIdx.x + threadIdx.x; //4
//int j = blockIdx.x;  //Data.count
int k = blockDim.y*blockIdx.y + threadIdx.y; //256
atomicAdd(&out[i], layer1[256*offset + k] * syn2[k*4 + i]);
}