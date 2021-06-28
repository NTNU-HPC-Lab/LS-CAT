#include "includes.h"
__global__ void initn(uint32_t *A, uint32_t size, uint32_t n) {
uint32_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
if(id < size) A[id] = n;
}