#include "includes.h"


#define BLOCK_SIZE 1024

#ifndef RADIUS
#define RADIUS 3
#endif

#ifndef ITERS
#define ITERS 100
#endif

#ifndef USE_L2
#define USE_L2 false
#endif





__global__ void stencil_no_shared(int *in, int *out)
{
int temp[BLOCK_SIZE + 2 * RADIUS];
int gindex = threadIdx.x + blockIdx.x * blockDim.x;
int lindex = threadIdx.x + RADIUS;
// Read input elements into shared memory
temp[lindex] = in[gindex+RADIUS];
if (threadIdx.x < RADIUS) {
temp[lindex - RADIUS] = in[gindex];
temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE + RADIUS];
}
__syncthreads();
// Apply the stencil
int result = 0;
for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
result += temp[lindex + offset];
// Store the result
out[gindex] = result;

}