#include "includes.h"
#define NOMINMAX








const unsigned int BLOCK_SIZE = 512;

__global__ void addKernel(float *c, const float *a, const float *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}