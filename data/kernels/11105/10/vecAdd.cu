#include "includes.h"
__global__ void vecAdd(float* d_A, float* d_B, float* d_C) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if(i<TAM)
d_C[i] = d_A[i] + d_B[i];
}