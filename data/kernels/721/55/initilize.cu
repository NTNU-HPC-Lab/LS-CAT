#include "includes.h"
__global__ void initilize( unsigned int seed, curandState_t *states)
{
int tid = threadIdx.x + blockDim.x * blockIdx.x;
curand_init(
seed,
tid,
0,
&states[tid]);
}