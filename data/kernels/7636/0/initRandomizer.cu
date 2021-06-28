#include "includes.h"
__global__ void initRandomizer(unsigned int seed, curandState* state){
int idx = blockIdx.x * blockDim.x + threadIdx.x;
curand_init(seed, idx, 0, &state[idx]);
}