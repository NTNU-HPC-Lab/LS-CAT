#include "includes.h"
__global__ void init_rng (curandState * states, int seed)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
curand_init (seed, tid, 0, &states[tid]);
}