#include "includes.h"
__global__ void sumArraysOnGPUshared(float *A, float *B, float *C, const int N)
{
__shared__ float smem[512];
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
smem[threadIdx.x] += i;
C[i] = A[i] + B[i] + smem[threadIdx.x];
}

}