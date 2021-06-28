#include "includes.h"
__global__ void init_random_seed(unsigned int seed, curandState_t *d_curand_state) {
int neuron = blockIdx.x*blockDim.x + threadIdx.x;
curand_init(seed, neuron, 0, &d_curand_state[neuron]);
}