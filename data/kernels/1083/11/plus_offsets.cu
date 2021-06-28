#include "includes.h"
__global__ void plus_offsets(float *coords, float *random, size_t total_size, float alpha){
size_t index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < total_size){
coords[index] += random[index] * alpha;
__syncthreads();
}
}