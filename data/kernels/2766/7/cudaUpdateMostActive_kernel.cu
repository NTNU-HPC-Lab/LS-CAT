#include "includes.h"
__global__ void cudaUpdateMostActive_kernel(unsigned int * exampleFiringRate, unsigned int * mostActiveId, unsigned int inputsDimX, unsigned int inputsDimY, unsigned int inputsDimZ)
{

const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

const unsigned int batchInputOffset = blockIdx.z * inputSize;

extern __shared__ unsigned int partialActiveIdx[];

// For case that threadIdx.x > inputSize
partialActiveIdx[threadIdx.x] = 0;

// TODO: Index 0 has a slight advantage here
for (unsigned int i=threadIdx.x; i<inputSize; i+=blockDim.x) {
partialActiveIdx[threadIdx.x] = threadIdx.x;
}

// Search for max ID in each thread
for (unsigned int i=threadIdx.x; i<inputSize; i+=blockDim.x) {
if (exampleFiringRate[i + batchInputOffset] >
exampleFiringRate[partialActiveIdx[threadIdx.x] + batchInputOffset]) {
partialActiveIdx[threadIdx.x] = i;
}
}

__syncthreads();

// Reduction over neurons
for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
if (threadIdx.x < offset){
if (exampleFiringRate[partialActiveIdx[threadIdx.x] + batchInputOffset] <
exampleFiringRate[partialActiveIdx[threadIdx.x + offset] + batchInputOffset]) {
partialActiveIdx[threadIdx.x] =
partialActiveIdx[threadIdx.x + offset];
}
}

__syncthreads();
}

if (threadIdx.x == 0) {
mostActiveId[blockIdx.z] = partialActiveIdx[0];
}

}