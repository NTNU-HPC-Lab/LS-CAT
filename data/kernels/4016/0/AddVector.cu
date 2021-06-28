#include "includes.h"

/* kernel.cu */




__global__ void AddVector( int vecSize, const float* vecA, const float* vecB, float* vecC)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < vecSize)
vecC[i] = vecA[i] + vecB[i];
}