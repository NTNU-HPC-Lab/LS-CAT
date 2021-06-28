#include "includes.h"
__global__ void Random( float *results, int n, unsigned int seed ) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
curandState_t state;

curand_init(seed, blockIdx.x, 0, &state);
results[ idx ] = curand(&state) / 1000.0f;
}