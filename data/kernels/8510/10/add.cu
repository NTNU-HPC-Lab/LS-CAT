#include "includes.h"
__global__ void add(int n, long *a, long *b, long *sum)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
sum[i] = a[i] + b[i];
printf("CUDA KERNEL ADD %ld + %ld = %ld \n",a[i],b[i],sum[i]);
}

}