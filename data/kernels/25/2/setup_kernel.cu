#include "includes.h"
__global__ void setup_kernel (curandState * state, unsigned long seed)
{
int id = blockIdx.x*NUM_BLOCKS + threadIdx.x;
curand_init ( seed, id, 0, &state[id] );
}