#include "includes.h"
__global__ void square_array(float *a, int array_size)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx<array_size) a[idx] = a[idx] * a[idx];
}