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

__device__ float maxOp(float a, float b) {
return a > b ? a : b;
}
__device__ float minOp(float a, float b) {
return a < b ? a : b;
}
__device__ float addOp(float a, float b) {
return a + b;
}
__global__ void reduceKernel(float* array, const size_t array_size, const unsigned int op, const size_t step)
{
__shared__ float temp[BLOCK_WIDTH];
int bx = blockIdx.x;
int tx = threadIdx.x;
int index = BLOCK_WIDTH * bx + tx;

if(index < array_size) {
temp[tx] = array[index * step];
}

__syncthreads();

// Reduce
for(int offset = BLOCK_WIDTH >> 1; offset > 0; offset >>= 1) {
if(tx < offset) {
switch(op) {
case ADD:
temp[tx] = addOp(temp[tx], temp[tx + offset]);
break;
case MIN:
temp[tx] = minOp(temp[tx], temp[tx + offset]);
break;
case MAX:
temp[tx] = maxOp(temp[tx], temp[tx + offset]);
break;
default:
break;
}
}
__syncthreads();
}

if(index < array_size) {
array[BLOCK_WIDTH * bx] = temp[0];
}

}