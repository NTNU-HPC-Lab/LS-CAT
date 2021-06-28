#include "includes.h"
__global__ void cudaUpdateFiringRate_kernel(int * firingRate, int * totalFiringRatePartial, unsigned int inputsDimX, unsigned int inputsDimY, unsigned int inputsDimZ)
{

const unsigned int inputSize = inputsDimZ * inputsDimX * inputsDimY;

const unsigned int batchInputOffset = blockIdx.z * inputSize;

const unsigned int blockOffset = blockIdx.x * blockDim.x;

const unsigned int partialIdx = threadIdx.x + blockOffset;

extern __shared__ unsigned int partialSum[];

// Perform first level of reduction during initialization
// This is more efficient since we need all threads to load data
// but the partial sum will see only half of the threads active
//partialSum[threadIdx.x] = firingRate[partialIdx + batchInputOffset] +
//    firingRate[partialIdx + blockDim.x + batchInputOffset];

partialSum[threadIdx.x] = 0;
if (partialIdx < inputSize){
partialSum[threadIdx.x] = firingRate[partialIdx + batchInputOffset];
}

__syncthreads();

// Reduction over neurons
for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
if (threadIdx.x < offset){
partialSum[threadIdx.x] += partialSum[threadIdx.x + offset];
}

__syncthreads();
}

if (threadIdx.x == 0) {
totalFiringRatePartial[blockIdx.x+gridDim.x*blockIdx.z] = partialSum[0];
}


}