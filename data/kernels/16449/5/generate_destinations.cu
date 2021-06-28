#include "includes.h"
__global__ void generate_destinations(curandState *state, int n, const uint32_t *sources, uint32_t *destinations) {
int first = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

curandState local_state = state[first];
for (int id = first ; id < n ; id += stride) {
destinations[id] = sources[curand(&local_state) % n];
}

state[first] = local_state;
}