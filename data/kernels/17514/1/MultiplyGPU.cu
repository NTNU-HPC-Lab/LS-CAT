#include "includes.h"



#define BLOCK_SIZE 16

__global__ void MultiplyGPU(float* a, float* b, float* c,int t)
{

int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

float aux =0;

if (i < t)
{
if (j < t)
{
for (int k = 0; k < t; k++)
{
aux += a[i * t + k] * b[k * t + j];
}

c[i * t + j] = aux;
}

}

}