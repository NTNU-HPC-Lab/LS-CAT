#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void initialise_curand_on_kernels(curandState* state, unsigned long seed) {
int idx = blockIdx.x*blockDim.x+threadIdx.x;
curand_init(seed, idx, 0, &state[idx]);
}