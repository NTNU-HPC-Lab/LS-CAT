#include "includes.h"
__global__ void FloatDiv(float *A, float *B, float *C)
{
unsigned int i = blockIdx.x * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;

if (B[i] != 0) {
C[i] = A[i] / B[i];
}
else {
C[i] = 0;
}

}