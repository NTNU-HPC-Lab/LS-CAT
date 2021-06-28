#include "includes.h"
__global__ void FloatMul(float *A, float *B, float *C)
{
unsigned int i = blockIdx.x * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
C[i] = A[i] * B[i];
}