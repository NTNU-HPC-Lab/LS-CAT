#include "includes.h"
__global__ void histo_kernel( unsigned char *buffer, long size, unsigned int *histo ) {
// calculate the starting index and the offset to the next
// block that each thread will be processing
int i = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while (i < size) {
atomicAdd( &histo[buffer[i]], 1 );
i += stride;
}
}