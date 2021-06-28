#include "includes.h"
__global__ void init_rand_states_kernel(curandState *state, int seed)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
/* Each thread gets same seed, a different sequence
number, no offset */
curand_init(seed, id, 0, &state[id]);
}