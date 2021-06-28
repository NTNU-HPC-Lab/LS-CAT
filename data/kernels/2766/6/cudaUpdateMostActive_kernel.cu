#include "includes.h"
__global__ void cudaUpdateMostActive_kernel(unsigned int * exampleIds, unsigned int * exampleFiringRate, unsigned int * mostActiveId, unsigned int inputsDimX, unsigned int inputsDimY, unsigned int inputsDimZ)
{

const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

const unsigned int batchInputOffset = blockIdx.z * inputSize;

const unsigned int blockOffset = blockIdx.x * blockDim.x;

const unsigned int partialIdx = threadIdx.x + blockOffset;

// TODO: Also used shared memory for firing rates to avoid global
// memory accesses
extern __shared__ unsigned int partialActiveIdx[];

// TODO: Index 0 has a slight advantage here
partialActiveIdx[threadIdx.x] = 0;
if (partialIdx < inputSize){
partialActiveIdx[threadIdx.x] = exampleIds[partialIdx + batchInputOffset];
}

__syncthreads();

// Reduction over neurons
for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
if (threadIdx.x < offset){
if (exampleFiringRate[partialActiveIdx[threadIdx.x]] <
exampleFiringRate[partialActiveIdx[threadIdx.x + offset]]) {
partialActiveIdx[threadIdx.x] =
partialActiveIdx[threadIdx.x + offset];
}
}

__syncthreads();
}

if (threadIdx.x == 0) {
mostActiveId[blockIdx.x+gridDim.x*blockIdx.z] = partialActiveIdx[0];
}

}