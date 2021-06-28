#include "includes.h"

#define MAX_VALUE 10


__global__ void saxpy(float *X, float *Y, float *Z, int A, int N)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if(i<N){
Z[i] = A * X[i] + Y[i];
}
}