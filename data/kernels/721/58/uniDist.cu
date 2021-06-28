#include "includes.h"
__global__ void uniDist( float *d_a, curandState_t *states, unsigned int size)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < size)
{
d_a[i] = curand_uniform(&states[i]);
}
}