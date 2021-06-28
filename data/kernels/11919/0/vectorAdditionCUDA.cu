#include "includes.h"

//no need for extern c according to stackoverflow answer by nvidia employee
extern "C"

__global__ void vectorAdditionCUDA(const float* a, const float* b, float* c, int n)
{
int ii = blockDim.x * blockIdx.x + threadIdx.x;
if (ii < n)
c[ii] = a[ii] + b[ii];
}