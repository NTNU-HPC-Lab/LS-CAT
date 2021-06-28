#include "includes.h"
__global__ void __randinit(unsigned long long seed, unsigned long long offset, curandState *rstates) {
int id = threadIdx.x + blockDim.x * blockIdx.x;
curand_init(seed, id, offset, &rstates[id]);
}