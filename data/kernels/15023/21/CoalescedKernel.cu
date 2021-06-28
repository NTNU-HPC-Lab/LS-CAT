#include "includes.h"
__global__ void CoalescedKernel(int *x, int *y, int *z, int *sum)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

sum[idx] = 0;
sum[idx] += x[idx] * x[idx];
sum[idx] += y[idx] * y[idx];
sum[idx] += z[idx] * z[idx];
}