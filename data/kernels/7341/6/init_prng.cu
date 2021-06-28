#include "includes.h"
__global__ void init_prng(curandState *states, const unsigned long int seed)
{
const int t_idx = threadIdx.x;
curand_init(seed, t_idx, 0, &states[t_idx]);
}