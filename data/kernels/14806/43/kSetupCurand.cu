#include "includes.h"
__global__ void kSetupCurand(curandState *state, unsigned long long seed) {
const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
/* Each thread gets same seed, a different sequence number,
no offset */
curand_init(seed, tidx, 0, &state[tidx]);
}