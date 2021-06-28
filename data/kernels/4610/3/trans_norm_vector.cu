#include "includes.h"
__global__ void trans_norm_vector(double* A, double* x, double* y, double* tmp, int NX, int NY)
{
int j;
int i = blockDim.x * blockIdx.x + threadIdx.x;

tmp[i] = 0;
//Α*Χ
for (j = 0; j < NY; j++) {
tmp[i] = tmp[i] + A[i*NY + j] * x[j];
}

}