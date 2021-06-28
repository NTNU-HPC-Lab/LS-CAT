#include "includes.h"
__global__ void innerProd(float *aa, float *bb, float *cc)
{
__shared__ float temp[THREADS_PER_BLOCK];
int index = threadIdx.x + blockIdx.x* blockDim.x;
temp[threadIdx.x] = aa[index]*bb[index];

*cc = 0; // Initialized to avoid memory problems. See comments
// below, next to the free and cudaFree commands.

// No thread goes beyond this point until all of them
// have reached it. Threads are only synchronized within
// a block.
__syncthreads();

//  Thread 0 sums the pairwise products
if (threadIdx.x == 0) {
float sum = 0;
for (int i = 0; i < THREADS_PER_BLOCK; i++){
sum += temp[i];
}
// Use atomicAdd to avoid different blocks accessing cc at the
// same time (race condition). The atomic opperation enables
// read-modify-write to be performed by a block without interruption.
//*cc += sum;
atomicAdd(cc, sum);
}

}