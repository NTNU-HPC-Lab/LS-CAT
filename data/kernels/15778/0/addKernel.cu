#include "includes.h"





// Helper function for using CUDA to add vectors in parallel.
__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}