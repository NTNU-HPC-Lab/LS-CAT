#include "includes.h"

__global__ void set_dynamic_indices(unsigned int *d_all_ib, unsigned int *d_ib, int flip)
{
if (flip == blockIdx.x)
{
d_ib[threadIdx.x] = d_all_ib[blockDim.x * blockIdx.x + threadIdx.x];
d_ib[3 + threadIdx.x] = 0;
}
}