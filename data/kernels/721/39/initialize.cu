#include "includes.h"
__global__ void initialize( unsigned int seed, curandState_t *states, unsigned int size)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < size)
{
curand_init(
seed,
i,
0,
&states[i]);
}
}