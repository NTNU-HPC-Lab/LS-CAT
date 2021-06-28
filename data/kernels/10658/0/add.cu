#include "includes.h"
extern "C"
__global__ void add(int n, float *a, float *b, float *sum)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
sum[i] = a[i] + 2*b[i];
}

}