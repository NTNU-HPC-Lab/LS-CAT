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
__global__ void one_stencil_with_rc (int *A, int *B, int sizeOfA)
{
// Declaring local register cache.
int rc[2];

// Id of thread in the warp.
int localId = threadIdx.x % WARP_SIZE;

// The first index of output element computed by this warp.
int startOfWarp = blockIdx.x * blockDim.x + WARP_SIZE*(threadIdx.x / WARP_SIZE);

// The Id of the thread in the scope of the grid.
int globalId = localId + startOfWarp;

if (globalId >= sizeOfA)
return;

// Fetching into shared memory.
rc[0] = A[globalId];
if (localId < 2 && WARP_SIZE + globalId < sizeOfA)
{
rc[1] =  A[WARP_SIZE + globalId];
}

// Each thread computes a single output.
int ac = 0;
int toShare = rc[0];

bool isLastWarp = sizeOfA - startOfWarp < WARP_SIZE;

// The number of threads in the warp which are inactive.
// Possibly bigger than zero only for the last warp.
int inactiveThreadsInWarp = isLastWarp ? startOfWarp + WARP_SIZE - sizeOfA : 0;


// Accessing register cache.
// We use a precomputed active mask.
// This is because otherwise only a subset of active threads return from
//	the __activemask() call, which will resemble a wrong picture of
//	the currently active threads in the warp.
//	notice that the active mask does not change along the following
//	loop so we claculate it just once.
//	Please refer to the cuda developers guide for futher information.
unsigned mask = //__activemask(); <-- Wrong!
(0xffffffff) >> (inactiveThreadsInWarp);

for (int i = 0 ; i < 3 ; ++i)
{
// Threads decide what value will be published in the following access.
if (localId < i)
toShare = rc[1];

ac += __shfl_sync(mask, toShare, (localId + i) % WARP_SIZE);
}

if (globalId < sizeOfA - 2)
B[globalId] = ac;
}