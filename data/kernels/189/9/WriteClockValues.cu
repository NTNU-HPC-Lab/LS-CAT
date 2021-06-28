#include "includes.h"
__global__ void WriteClockValues( unsigned int *completionTimes, unsigned int *threadIDs )
{
size_t globalBlock = blockIdx.x+blockDim.x*(blockIdx.y+blockDim.y*blockIdx.z);
size_t globalThread = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);

size_t totalBlockSize = blockDim.x*blockDim.y*blockDim.z;
size_t globalIndex = globalBlock*totalBlockSize + globalThread;

completionTimes[globalIndex] = clock();
threadIDs[globalIndex] = threadIdx.y<<4|threadIdx.x;
}