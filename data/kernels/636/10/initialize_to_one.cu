#include "includes.h"
__global__ void initialize_to_one( uint32_t *reduction, uint32_t size)
{
uint32_t t_index = blockDim.x * blockIdx.x + threadIdx.x;
if(t_index < size) {
reduction[t_index] = 1;
}
}