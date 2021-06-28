#include "includes.h"
__global__ void simple_corner_turn_kernel(float *d_input, float *d_output, int primary_size, int secondary_size){

size_t primary = blockIdx.x * blockDim.x + threadIdx.x;
size_t secondary = blockIdx.y * blockDim.y + threadIdx.y;

d_output[(size_t)primary*secondary_size + secondary] = (float) __ldg(&d_input[(size_t)secondary*primary_size + primary]);
}