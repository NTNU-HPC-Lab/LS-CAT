#include "includes.h"

// Number of elements to put in the test array
#define TEST_SIZE 16
#define NUM_BINS 10

////////////////////////////////////////////////////////////////
////////////////// COPY EVERYTHING BELOW HERE //////////////////
////////////////////////////////////////////////////////////////

// Number of threads per block (1-d blocks)
#define BLOCK_WIDTH 4
// Functions to reduce with
#define ADD 0
#define MIN 1
#define MAX 2
// Device functions

__global__ void scanKernel(unsigned int* d_cdf, unsigned int* d_input, const size_t array_size)
{
__shared__ unsigned int temp[BLOCK_WIDTH<<1];
int bx = blockIdx.x;
int tx = threadIdx.x;
int index = BLOCK_WIDTH * bx + tx;
int offset = 1;

if(2*index + 1 < array_size) {
temp[2*index] = d_input[2*index];
temp[2*index + 1] = d_input[2*index + 1];
}

// Up-sweep
for(int powOf2 = (2*BLOCK_WIDTH)>>1; powOf2 > 0; powOf2 >>= 1) {
__syncthreads();
if(tx < powOf2) {
int idx1 = offset*(2*tx + 1) - 1 + 2*BLOCK_WIDTH*bx;
int idx2 = offset*(2*tx + 2) - 1 + 2*BLOCK_WIDTH*bx;
temp[idx2] += temp[idx1];
}
offset <<= 1;
}

__syncthreads();
///// The below will need to be remembered for multiple blocks /////
if(tx == 0) {
temp[2*BLOCK_WIDTH*(bx + 1) - 1] = 0;
}

// Down-sweep
for(int powOf2 = 1; powOf2 < 2*BLOCK_WIDTH; powOf2 <<= 1) {
offset >>= 1;
__syncthreads();
if(tx < powOf2) {
int idx1 = offset*(2*tx + 1) - 1 + 2*BLOCK_WIDTH*bx;
int idx2 = offset*(2*tx + 2) - 1 + 2*BLOCK_WIDTH*bx;
unsigned int t = temp[idx1];
temp[idx1] = temp[idx2];
temp[idx2] += t;
}
}

__syncthreads();
if(2*index + 1 < array_size) {
d_cdf[2*index] = temp[2*index];
d_cdf[2*index + 1] = temp[2*index + 1];
}
}