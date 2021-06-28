#include "includes.h"
__global__ void total(float * input, float * output, int len) {
//@@ Load a segment of the input vector into shared memory
__shared__ float partialSum[2 * BLOCK_SIZE];
unsigned int tx = threadIdx.x;
unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

if ((start + tx) < len) {
partialSum[tx] = input[start + tx];
}
else {
partialSum[tx] = 0.0;
}
if ((start + BLOCK_SIZE + tx) < len) {
partialSum[BLOCK_SIZE + tx] = input[start + BLOCK_SIZE + tx];
}
else {
partialSum[BLOCK_SIZE + tx] = 0.0;
}

//@@ Traverse the reduction tree
for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride/=2) {
__syncthreads();
if (tx < stride) {
partialSum[tx] += partialSum[tx + stride];
}
}

//@@ Write the computed sum of the block to the output vector at the
//@@ correct index
// Boundary condition is handled by filling “identity value (0 for sum)”
// into the shared memory of the last block
if (tx == 0) {
output[blockIdx.x] = partialSum[0];
}

}