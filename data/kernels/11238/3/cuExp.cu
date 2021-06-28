#include "includes.h"
extern "C"

extern "C"

extern "C"

extern "C"

__global__ void cuExp(int n, float *a, float *result)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
result[i] = expf(a[i]);
}

}