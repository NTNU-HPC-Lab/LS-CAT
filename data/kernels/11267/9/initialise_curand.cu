#include "includes.h"

__global__ void initialise_curand(curandState * state, unsigned long seed)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
curand_init(seed, idx, 0, &state[idx]);
//printf("index idx = %d", idx);
__syncthreads();
}