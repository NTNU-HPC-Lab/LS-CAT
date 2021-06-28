#include "includes.h"



enum ComputeMode { ADD, SUB, MUL, DIV };
cudaError_t computeWithCuda(int *c, const int *a, const int *b, unsigned int size, ComputeMode mode);

__global__ void compareWithOneKernel(float* b, const double* a)
{
int i = threadIdx.x;
if(a[i] == 1)
b[i] = b[i] + 1;
}