#include "includes.h"
__global__ void parallelMeanUnroll2(float* d_inputArray, uint64_t inputLength, float* d_outputMean)
{
uint32_t localThreadIndex = threadIdx.x;
uint32_t sumDataIndex = blockIdx.x * blockDim.x * 2 + localThreadIndex; //The index of the piece of data that I will sum into this current block
uint32_t globalThreadIndex = blockDim.x * blockIdx.x + localThreadIndex;

//calculate a pointer to this threadBlocks data
float* localBlockPointer = d_inputArray + blockIdx.x * blockDim.x * 2;

//Add the next blockDim.x's worth of data into this block before we start reducing
//Bounds checking
if(sumDataIndex + blockDim.x < inputLength)
{
d_inputArray[sumDataIndex] += d_inputArray[sumDataIndex + blockDim.x];
}

//Wait for all threads on this block to complete
__syncthreads();

//Start reducing
//In-place, strided, reduction
for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
{
if (localThreadIndex < stride)
{
localBlockPointer[localThreadIndex] += localBlockPointer[localThreadIndex + stride];
}
}

//Wait for all threads on this block to complete
__syncthreads();

//If this is the thread with the global index of one, calculate the mean
if(globalThreadIndex == 0)
{
//Clear the output just incase it isn't already
*d_outputMean = 0;

for(uint32_t i = 0; i < gridDim.x; ++i)
{
*d_outputMean += d_inputArray[ i * blockDim.x * 2]; //Times 2 because we take in 'two blocks' worth of data for each actual block
}

*d_outputMean =  *d_outputMean / (inputLength - 1);

//printf("Mean: %f\n", *d_outputMean);
}

}