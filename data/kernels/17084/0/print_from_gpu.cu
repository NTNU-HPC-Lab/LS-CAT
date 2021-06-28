#include "includes.h"
__global__ void print_from_gpu(void) {
int tidx = blockIdx.x*blockDim.x+threadIdx.x;
printf("Hello from device! My threadId = blockIdx.x *blockDim.x + threadIdx.x <=> %d = %d * %d + %d \n",
tidx, blockIdx.x, blockDim.x, threadIdx.x);
}