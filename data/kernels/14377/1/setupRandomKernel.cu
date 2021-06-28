#include "includes.h"
__global__ void setupRandomKernel(curandState* states, unsigned long long seed) {
unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
curand_init(seed, tid, 0, &states[tid]);
}