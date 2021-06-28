#include "includes.h"

/**
* @brief cudaCreateBuffer Allocates a cuda buffer and stops the programm on error.
* @param size
* @return
*/
__global__ void kernelSetDoubleBuffer(float* gpuBuffPtr, float v, size_t size)
{
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index < size)
gpuBuffPtr[index] = v;
}