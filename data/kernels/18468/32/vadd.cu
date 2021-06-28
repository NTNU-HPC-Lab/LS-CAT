#include "includes.h"
__global__ void vadd(const float *A, const float *B, float *C, int ds){

for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x)         // a grid-stride loop
C[idx] = A[idx] + B[idx]; // do the vector (element) add here
}