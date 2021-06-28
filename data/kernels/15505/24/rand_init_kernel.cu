#include "includes.h"
__global__ void rand_init_kernel(int seed, curandStatePhilox4_32_10_t *states, int n) {
int x(threadIdx.x + blockDim.x * blockIdx.x);

if (x < n)
curand_init(seed, x, 0, &states[x]);
}