#include "includes.h"
__global__ void setup_kernel ( curandState *state, unsigned long seed )
{
curand_init ( seed, 0, 0, &state[0] );
}