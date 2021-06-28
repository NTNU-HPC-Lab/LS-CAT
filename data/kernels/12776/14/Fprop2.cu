#include "includes.h"
__global__ void Fprop2(const float* layer1, const float* syn2, float* out)
{
int i = blockDim.y*blockIdx.y + threadIdx.y; //10
int j = blockIdx.x;  //Data.count
//int k = threadIdx.x; //256
float x = 0.0;
for (int k=0; k < 256; ++k)
x += layer1[j*256 + k] * syn2[k*10 + i];
out[j*10 + i] = x;
}