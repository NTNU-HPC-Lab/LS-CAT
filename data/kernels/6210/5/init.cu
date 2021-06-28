#include "includes.h"
__global__ void init(uint32_t *v, uint32_t val, uint32_t n) {
auto i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n)
v[i] = val;
if (i == 0)
printf("init\n");
}