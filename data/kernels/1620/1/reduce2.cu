#include "includes.h"

// CUDA runtime

// Helper functions and utilities to work with CUDA

//Standard C library

#define subCOL 5248
#define COL 5248
#define ROW 358
#define WARPABLEROW 512
#define blocksize 256
#define subMatDim subCOL*WARPABLEROW
#define targetMatDim ROW * COL
__global__ void reduce2(int *g_idata, int *g_odata, int g_size)
{
__shared__ int sdata[blocksize];
// each thread loads one element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
sdata[tid] = g_idata[i];
__syncthreads();
// do reduction in shared mem
for (unsigned int s = 1; s < blockDim.x; s *= 2)
{
int index = 2 * s*tid;

if (index < blockDim.x)
{
sdata[index] += sdata[index + s];
}

__syncthreads();
}
// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}