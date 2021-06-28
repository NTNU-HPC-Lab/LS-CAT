#include "includes.h"

static char* program_name;

// Usage
__global__ void jacobiOnDevice(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
{
float sigma = 0.0;
int idx = threadIdx.x;
for (int j=0; j<Nj; j++)
{
if (idx != j)
sigma += A[idx*Nj + j] * x_now[j];
}
x_next[idx] = (b[idx] - sigma) / A[idx*Nj + idx];
}