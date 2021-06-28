#include "includes.h"
__global__ void cudaSetupRng_kernel(curandState * state, unsigned int seed)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
// Each thread gets the same seed but a different sequence number
curand_init(seed, id, 0, &state[id]);
}