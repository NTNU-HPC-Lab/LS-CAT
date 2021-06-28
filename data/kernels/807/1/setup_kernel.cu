#include "includes.h"
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
int id = blockIdx.x;
int sequence = id;
int offset = 0;
curand_init ( seed, sequence, offset, &state[id] );
}