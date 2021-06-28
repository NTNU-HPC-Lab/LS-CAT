#include "includes.h"
__global__ void stencil_no_sync(int *in, int *out)
{
__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
int gindex = threadIdx.x + blockIdx.x * blockDim.x;
int lindex = threadIdx.x + RADIUS;
// Read input elements into shared memory
temp[lindex] = in[gindex+RADIUS];
if (threadIdx.x < RADIUS) {
temp[lindex - RADIUS] = in[gindex];
temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE + RADIUS];
}
////////////////////////////// missing sync thread ////////////////////////

// Apply the stencil
int result = 0;
for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
result += temp[lindex + offset];
// Store the result
out[gindex] = result;

}