#include "includes.h"
__global__ void _bcnn_vsum_kernel(int n, float *x, float *sum)
{
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n)
*sum += x[i];
}