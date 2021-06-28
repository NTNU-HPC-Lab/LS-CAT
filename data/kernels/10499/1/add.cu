#include "includes.h"
extern "C"
__global__ void add(int n, float *a, float *sum)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i<n)
{
for (int j = 0; j < n; j++)
{
sum[i] = sum[i] + a[i*n + j];
}
}

}