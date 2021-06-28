#include "includes.h"
/////////////////////////////////////////////////////////

// Computes the 1-stencil using GPUs.
// We don't check for error here for brevity.
// In your implementation - you must do it!

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

#ifndef k
#define k 3
#endif
#ifndef OUTPUT_PER_THREAD
#define OUTPUT_PER_THREAD 1
#endif
#define LOCAL_REGISTER_SIZE ((1+OUTPUT_PER_THREAD) > (k+31)/32 ? (1+OUTPUT_PER_THREAD) : (k+31)/32)
#ifndef TEST_TIMES
#define TEST_TIMES 5
#endif

float host_k_stencil (int *A, int *B, int sizeOfA, int withRc);
__global__ void one_stencil (int *A, int *B, int sizeOfA)
{
extern __shared__ int s[];
// Id of thread in the block.
int localId = threadIdx.x;

// The first index of output element computed by this block.
int startOfBlock = blockIdx.x * blockDim.x;

// The Id of the thread in the scope of the grid.
int globalId = localId + startOfBlock;

if (globalId >= sizeOfA)
return;

// Fetching into shared memory.
s[localId] = A[globalId];
if (localId < 2 && blockDim.x + globalId < sizeOfA)
{
s[blockDim.x + localId] =  A[blockDim.x + globalId];
}

// We must sync before reading from shared memory.
__syncthreads();

// Each thread computes a single output.
if (globalId < sizeOfA - 2)
B[globalId] = s[localId] + s[localId + 1] + s[localId + 2];
}