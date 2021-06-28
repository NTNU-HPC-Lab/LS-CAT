#include "includes.h"
__global__ void init_randoms(unsigned int seed, curandState_t* states) {
int index = blockDim.x * blockIdx.x + threadIdx.x;

curand_init(seed, index, 0, &states[index]);
}