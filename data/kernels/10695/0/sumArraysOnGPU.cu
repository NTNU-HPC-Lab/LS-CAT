#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
int i=blockIdx.x*COL+threadIdx.x;
//printf("[gpu]:gridDim.x=%u, gridDim.y=%u, gridDim.z=%u, blockDim.x=%u, blockDim.y=%u, blockDim.z=%u, blockIdx.x=%u, blockIdx.y=%u, blockIdx.z=%u,threadIdx.x=%u, threadIdx.y=%u, threadIdx.z=%u\n",
//gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z,threadIdx.x, threadIdx.y, threadIdx.z);
C[i]=A[i]+B[i];
//printf("sum[%u][%u]: A[%5.5f]+B[%5.5f]=C[%5.5f]\n",blockIdx.x, threadIdx.x, A[i], B[i], C[i]);
}