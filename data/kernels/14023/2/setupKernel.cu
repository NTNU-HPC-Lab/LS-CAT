#include "includes.h"
__global__ void setupKernel(curandState *state, unsigned long long seed) {
int idx = threadIdx.x + blockDim.x * blockIdx.x;
curand_init(seed, idx, 0, &state[idx]);
}