#include "includes.h"



enum ComputeMode { ADD, SUB, MUL, DIV };
cudaError_t computeWithCuda(int *c, const int *a, const int *b, unsigned int size, ComputeMode mode);

__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}