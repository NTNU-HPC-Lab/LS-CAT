#include "includes.h"



__global__ void matrixVectMultKernel(float* A, float* B, float* C, int n)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
int elementPos = i;
i = i * n;
int limit = i + n;
int j = 0;

if(i < n*n){
C[elementPos] = 1;

while((i < limit) && (j < n)){
C[elementPos] = C[elementPos] * (A[i]+B[j]);
i++;
j++;
}
}
}