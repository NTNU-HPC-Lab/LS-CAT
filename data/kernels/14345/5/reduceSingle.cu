#include "includes.h"
__global__ void reduceSingle(int *idata, int *single, int nrows)
{
// Reduce rows to the first element in each row
int i;
extern __shared__ int parts[];

// Each block gets a row, each thread will reduce part of a row

// Calculate our offset into the row
// The number of cols per thread

// Sum my part of one dimensional array and put it shared memory
parts[threadIdx.x] = 0;
for (i = threadIdx.x; i < nrows; i+=blockDim.x) {
parts[threadIdx.x] += idata[i];
}
int tid = threadIdx.x;
if (tid < 512) { parts[tid] += parts[tid + 512];}
__syncthreads();
if (tid < 256) { parts[tid] += parts[tid + 256];}
__syncthreads();
if (tid < 128) { parts[tid] += parts[tid + 128];}
__syncthreads();
if (tid < 64) { parts[tid] += parts[tid + 64];}
__syncthreads();
if (tid < 32) { parts[tid] += parts[tid + 32];}
__syncthreads();
if(threadIdx.x == 0) {
*single = 0;
for(i = 0; i < 32; i++) {
*single += parts[i];
}
}
}