#include "includes.h"
__global__ void initMult(uint32_t *A, uint32_t size, uint32_t mult) {
uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
if(id < size) A[id] = id * mult;
}