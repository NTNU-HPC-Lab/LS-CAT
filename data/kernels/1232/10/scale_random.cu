#include "includes.h"
__global__ void scale_random(float *random, size_t total_size){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < total_size){
random[index] = random[index] * 2.0 - 1.0;
__syncthreads();
}
}