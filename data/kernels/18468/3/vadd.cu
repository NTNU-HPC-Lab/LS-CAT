#include "includes.h"
__global__ void vadd(const float *A, const float *B, float *C, int ds){

int idx = threadIdx.x+blockDim.x*blockIdx.x;
if (idx < ds)
C[idx] = A[idx] + B[idx];
}