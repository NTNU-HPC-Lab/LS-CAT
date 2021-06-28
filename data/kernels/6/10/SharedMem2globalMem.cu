#include "includes.h"
__global__ void SharedMem2globalMem(float * d_memoryA, int iSize)
{
/* Amount of shared memory is determined by host call */
extern __shared__ float s_memoryA[];
/* Generate global index */
int iID = blockDim.x * blockIdx.x + threadIdx.x;
/* Get the number of available threads */
int iNumThreads = blockDim.x * gridDim.x;
/* Calculate number of elements */
int iNumElements = iSize / sizeof(float);
/* Read global memory (coalesce) to shared memory */
for(int i = iID; i < iNumElements; i += iNumThreads)
d_memoryA[i] = s_memoryA[i];
}