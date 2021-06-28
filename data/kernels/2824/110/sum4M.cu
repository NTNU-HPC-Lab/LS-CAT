#include "includes.h"
__global__ void sum4M(float *A, float *B, float *C, const int N)
{
int j;
int i = blockIdx.x * blockDim.x + threadIdx.x;
float x[4];

#pragma unroll
for (j=0; j < 4; j++)
if  (i < N) {
x[j] = A[i]/7.0*A[i];
C[i] += A[i]/3 + 17*B[i] + 3*B[i] - A[i]*x[j] + x[j]*B[i]*7;
i += blockDim.x * gridDim.x;
}
}