#include "includes.h"
__global__ void stencil_ld(unsigned *in, unsigned *out){
printf("Thread %d : %d\n", threadIdx.x, in[threadIdx.x]);
out[threadIdx.x] = 2 * in[threadIdx.x];
printf("out location : %p\n", out+threadIdx.x);
printf("in %d : %d\n" , threadIdx.x, in[threadIdx.x]);
printf("out %d : %d\n", threadIdx.x, out[threadIdx.x]);

__syncthreads();

/*
__shared__ int temp[BLOCK_SIZE + 2*RADIUS];
int gindex = threadIdx.x + blockIdx.x * blockDim.x;
int lindex = threadIdx.x + RADIUS;

temp[lindex] = in[gindex];

if(threadIdx.x < RADIUS){
temp[lindex - RADIUS]     = in[gindex - RADIUS];
temp[lindex + BLOCK_SIZE] = in[gindex - BLOCK_SIZE];
}

int result = 0;
for(int offset = -RADIUS; offset < RADIUS; offset++){
result += temp[lindex + offset];
}

__syncthreads();

out[gindex] = result;
*/
}