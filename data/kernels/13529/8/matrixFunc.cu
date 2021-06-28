#include "includes.h"
__global__ void matrixFunc(float *F, int size)
{
#pragma unroll 16
for(int k = 0; k < 100; k++)
#pragma unroll 16
for(int i = 1; i < size; i++)
for(int j = 0; j < size - 1; j++)
F[i * size + j] = F[(i-1) * size + j + 1] + F[i * size + j + 1];
}