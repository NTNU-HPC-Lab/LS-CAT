#include "includes.h"
__global__ void reduceSum(int *idata, int *odata, unsigned int ncols)
{
// Reduce rows to the first element in each row
int i;
int blockOffset;
int rowStartPos;
int colsPerThread;
int *mypart;

// Each block gets a row, each thread will reduce part of a row

// Calculate the offset of the row
blockOffset = blockIdx.x * ncols;
// Calculate our offset into the row
rowStartPos = threadIdx.x * (ncols/blockDim.x);
// The number of cols per thread
colsPerThread = ncols/blockDim.x;

mypart = idata + blockOffset + rowStartPos;

// Sum all of the elements in my thread block and put them
// into the first column spot
for (i = 1; i < colsPerThread; i++) {
mypart[0] += mypart[i];
}
__syncthreads(); // Wait for everyone to complete
// Now reduce all of the threads in my block into the first spot for my row
if(threadIdx.x == 0) {
odata[blockIdx.x] = 0;
for(i = 0; i < blockDim.x; i++) {
odata[blockIdx.x] += mypart[i*colsPerThread];
}
}
// We cant synchronize between blocks, so we will have to start another kernel
}