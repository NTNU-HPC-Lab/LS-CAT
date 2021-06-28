#include "includes.h"



#define BLOCK_SIZE 16

__device__ float* GetSubMatrix(float * a, int tam, int row, int col)
{
float* aSub;
aSub = &a[tam * BLOCK_SIZE * row + BLOCK_SIZE * col];
return aSub;
}
__global__ void MultiplyGPUMult(float * a, float *b, float *c,int t)
{
int blockRow = blockIdx.y;
int blockCol = blockIdx.x;

float* Csub = GetSubMatrix(c, t, blockRow, blockCol);

float Cvalue = 0;

int row = threadIdx.y;
int col = threadIdx.x;

for (int m = 0; m < t / BLOCK_SIZE; m++)
{
float* Asub = GetSubMatrix(a, t, blockRow, m);
float* Bsub = GetSubMatrix(b, t, m, blockCol);

__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

As[row][col] = Asub[row * t + col];
Bs[row][col] = Bsub[row * t + col];

__syncthreads();

for (int e = 0; e < BLOCK_SIZE; e++)
{
Cvalue += As[row][e] * Bs[e][col];
}

__syncthreads();

}

Csub[row * t + col] = Cvalue;
}