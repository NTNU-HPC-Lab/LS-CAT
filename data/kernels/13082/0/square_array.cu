#include "includes.h"
__global__ void square_array(float *a, int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx<N) a[idx] = a[idx] * a[idx];
}