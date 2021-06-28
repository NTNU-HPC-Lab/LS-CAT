#include "includes.h"
__global__ void stencil_ld(unsigned *in, unsigned *out){
__shared__ int temp[BLOCK_SIZE + 2*RADIUS];
int gindex = threadIdx.x + blockIdx.x * blockDim.x;
int lindex = threadIdx.x;

temp[lindex] = in[gindex];

if(threadIdx.x < RADIUS){
temp[lindex - RADIUS]     = in[gindex - RADIUS];
temp[lindex + BLOCK_SIZE] = in[gindex - BLOCK_SIZE];
}


__syncthreads();

int result = 0;
for(int offset = -RADIUS; offset < RADIUS; offset++){
result += temp[lindex + offset];
}


out[gindex] = result;

}