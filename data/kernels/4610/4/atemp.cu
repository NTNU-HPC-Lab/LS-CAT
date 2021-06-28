#include "includes.h"
__global__ void atemp(double* A, double* y, double* tmp, int NX, int NY)
{
int j;
int i = blockDim.x * blockIdx.x + threadIdx.x;
// Α(T)*temp
if (i <= NY){
for (j = 0; j < NX; j++) {
y[i] = y[i] + A[i + j*NY] * tmp[j];
}
}
}