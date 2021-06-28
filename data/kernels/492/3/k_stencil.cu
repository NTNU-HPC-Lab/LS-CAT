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
__global__ void k_stencil (int *A, int *B, int sizeOfA)
{
extern __shared__ int s[];
// Id of thread in the block.
int localId = threadIdx.x;

// The first index of output element computed by this block.
int startOfBlock = blockIdx.x * blockDim.x * OUTPUT_PER_THREAD;

// The Id of the thread in the scope of the grid.
int globalId = localId + startOfBlock;

if (globalId >= sizeOfA)
return;

// Fetching into shared memory.
for (int i = 0 ; i < OUTPUT_PER_THREAD ; ++i)
{
if (globalId + i*BLOCK_SIZE < sizeOfA)
{
s[localId + i*BLOCK_SIZE] = A[globalId + i*BLOCK_SIZE];
}
}

if (localId < k && blockDim.x*OUTPUT_PER_THREAD + globalId < sizeOfA)
{
s[localId + blockDim.x*OUTPUT_PER_THREAD] =  A[blockDim.x*OUTPUT_PER_THREAD + globalId];
}

// We must sync before reading from shared memory.
__syncthreads();

int sum = 0;
for (int j = 0 ; j < OUTPUT_PER_THREAD ; ++j)
{
sum = 0;
if (globalId + j*BLOCK_SIZE >= sizeOfA - k)
return;
for (int i = 0 ; i < k + 1 ; ++i)
{
sum += s[localId + j*BLOCK_SIZE + i];
}
B[globalId + BLOCK_SIZE*j] = sum ;
}
}