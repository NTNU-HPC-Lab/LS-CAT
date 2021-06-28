#include "includes.h"
__global__ void reduction(int * in, int * out){
int globalid = blockIdx.x*blockDim.x + threadIdx.x;
__shared__ int s_array[BLOCK_DIM];

s_array[threadIdx.x] = in[globalid];
__syncthreads();

for (int i = 1; i < blockDim.x; i *= 2){
if (threadIdx.x % (2*i) == 0){
s_array[threadIdx.x] += s_array[threadIdx.x+i];
}
__syncthreads();
}

if (threadIdx.x == 0)
out[blockIdx.x] = s_array[0];
}