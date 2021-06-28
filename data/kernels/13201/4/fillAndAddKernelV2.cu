#include "includes.h"
#define NOMINMAX








const unsigned int BLOCK_SIZE = 512;

__global__ void fillAndAddKernelV2(float* c, float *a, float* b)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
a[i] = sin((double)i)*sin((double)i);
b[i] = cos((double)i)*cos((double)i);
c[i] = a[i] + b[i];
}