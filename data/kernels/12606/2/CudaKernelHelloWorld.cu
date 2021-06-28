#include "includes.h"
__global__ void CudaKernelHelloWorld(char *a, int *b)
{
a[threadIdx.x] += b[threadIdx.x];
}