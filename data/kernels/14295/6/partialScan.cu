#include "includes.h"
__global__ void partialScan(unsigned int *d_in, unsigned int *d_out, unsigned int *d_total, size_t n)
{
__shared__ unsigned int temp[BLOCK_WIDTH];
int tx = threadIdx.x;
int bx = blockIdx.x;
int index = BLOCK_WIDTH * bx + tx;

if(index < n) {
temp[tx] = d_in[index];
} else { temp[tx] = 0; }
__syncthreads();

// Perform the actual scan
for(int offset = 1; offset < BLOCK_WIDTH; offset <<= 1) {
if(tx + offset < BLOCK_WIDTH) {
temp[tx + offset] += temp[tx];
}
__syncthreads();
}

// Shift when copying the result so as to make it an exclusive scan
if(tx +1 < BLOCK_WIDTH && index + 1 < n) {
d_out[index + 1] = temp[tx];
}
d_out[0] = 0;

// Store the total sum of each block
d_total[bx] = temp[BLOCK_WIDTH - 1];
}