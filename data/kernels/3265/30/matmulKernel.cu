#include "includes.h"
__global__ void matmulKernel(float *A, float *B, float *C, int rA, int cA, int cB){
int i = blockIdx.y*gridDim.x + blockIdx.x, j = threadIdx.y*blockDim.x + threadIdx.x;
if(i < rA && j < cB){
C[i*cB + j] = 0.;
for(int k=0;k<cA;++k) C[i*cB + j] += A[i*cA + k] * B[k*cB + j];
}
return;
}