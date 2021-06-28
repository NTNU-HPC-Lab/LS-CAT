#include "includes.h"
__global__ void matMul(double *a, double *b, double *c)
{
const int NUM_THREAD_IN_BLOCK = blockDim.x * blockDim.y * blockDim.z;

int bID = blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK) + blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z));
int tID = bID + ((blockDim.y * blockDim.x) * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;


for(int i = 0; i < MATRIX_J; i++)
c[tID] += a[(tID * MATRIX_J) + i] * b[i];
}