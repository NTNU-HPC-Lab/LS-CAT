#include "includes.h"



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
c[i] = a[i] + b[i];
}