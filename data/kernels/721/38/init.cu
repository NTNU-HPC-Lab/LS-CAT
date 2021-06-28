#include "includes.h"
__global__ void init( unsigned int seed, curandState_t *d_state)
{
curand_init(
seed,
threadIdx.x + blockDim.x * blockIdx.x,
0,
&d_state[threadIdx.x + blockDim.x * blockIdx.x]);
}