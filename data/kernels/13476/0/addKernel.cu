#include "includes.h"




cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


const int arraySize = 10000000;

__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}