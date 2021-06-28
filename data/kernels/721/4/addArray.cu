#include "includes.h"
__global__ void addArray( float *d_a, float *d_b, float *d_c, int size)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i >= size)
{
return;
}
d_c[i] = d_a[i] + d_b[i];
}