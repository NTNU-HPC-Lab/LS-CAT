#include "includes.h"
__global__ void histo_kernel(unsigned char *buffer1, long size1, unsigned int *histo1){
int i = threadIdx.x + blockDim.x * blockIdx.x;
int stride = blockDim.x * gridDim.x;

while (i < size1){
atomicAdd(&(histo1[buffer1[i]]),1);
i += stride;
}
}