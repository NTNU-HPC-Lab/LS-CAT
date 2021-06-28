#include "includes.h"
__global__ void Fprop1(const float* in, const float* syn1, float* layer1)
{
int i = threadIdx.x;                         //256
//int j = blockDim.y*blockIdx.y + threadIdx.y; //28*28
int k = blockIdx.x;                          //Data.count
float x = 0.0;
for (int j=0; j < 28*28; ++j)
x += in[k*28*28 + j] * syn1[j*256 + i];
layer1[k*256 + i] = x;
}