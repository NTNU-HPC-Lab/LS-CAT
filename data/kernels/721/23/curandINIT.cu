#include "includes.h"
__global__ void curandINIT( int size, curandState_t *d_states, unsigned int seed)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < size)
{
curand_init(seed, i, 0, &d_states[i]);
}
}