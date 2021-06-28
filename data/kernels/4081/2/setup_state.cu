#include "includes.h"
__global__ void setup_state(curandState* state, unsigned long long seed) {
curand_init(seed, 0, 0, state);
}