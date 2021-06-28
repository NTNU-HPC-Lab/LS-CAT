#include "includes.h"


typedef float dtype;

#define N_ (8 * 1024 * 1024)
#define MAX_THREADS 256	   // threads per block
#define MAX_BLOCKS 64

#define MIN(x,y) ((x < y) ? x : y)


/* return the next power of 2 number that is larger than x */
__global__ void kernel5(dtype *g_idata, dtype *g_odata, unsigned int n)
{
__shared__  volatile dtype scratch[MAX_THREADS];
unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
unsigned int blockDimNew = blockDim.x * 2; 			// since the new blockDim will be halved for the loop iterations
unsigned int i = (bid * blockDimNew) + threadIdx.x;
unsigned int gridSize = blockDim.x * 2 * gridDim.x;

// each thread sums up 512 elements before storing into shared array
scratch[threadIdx.x] = 0;
while(i < n) {
scratch[threadIdx.x] += g_idata[i] + g_idata[i + blockDim.x];
i += gridSize;	//stride length
}
__syncthreads ();

int warp_size = 32;

for(int stride = (blockDim.x/2); stride > warp_size; stride = (stride/2)) {	//repeat until stride is 32 (one warp left at this point and no active threads)

if(threadIdx.x < stride) {				// check index range
scratch[threadIdx.x] += scratch[threadIdx.x + stride];
}
__syncthreads ();
}

// manually reduce
if(threadIdx.x <= warp_size)
{
scratch[threadIdx.x] += scratch[threadIdx.x + warp_size];
scratch[threadIdx.x] += scratch[threadIdx.x + warp_size/2];
scratch[threadIdx.x] += scratch[threadIdx.x + warp_size/4];
scratch[threadIdx.x] += scratch[threadIdx.x + warp_size/8];
scratch[threadIdx.x] += scratch[threadIdx.x + warp_size/16];
scratch[threadIdx.x] += scratch[threadIdx.x + 1];
}
__syncthreads ();

if(threadIdx.x == 0) {		// copy back to global array
g_odata[bid] = scratch[0];
}


}