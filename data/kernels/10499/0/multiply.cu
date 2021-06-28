#include "includes.h"
extern "C"
__global__ void multiply(int n, float *a, float *b, float *sum)
{
int ind = threadIdx.x + blockDim.x * blockIdx.x;
int i = ind ;
int j = ind % n;

if (j<n)
{
sum[i] = a[i] * b[j];
}

}