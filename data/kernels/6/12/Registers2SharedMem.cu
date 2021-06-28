#include "includes.h"
__global__ void Registers2SharedMem(float * outFloat, int iSize)
{
/* Amount of shared memory is determined by host call */
extern __shared__ float s_memoryA[];
/* Variable in register */
float r_var;
/* Generate global index */
int iID = blockDim.x * blockIdx.x + threadIdx.x;
/* Get the number of available threads */
int iNumThreads = blockDim.x * gridDim.x;
/* Calculate number of elements */
int iNumElements = iSize / sizeof(float);
/* Read global memory (coalesce) to shared memory */
for(int i = iID; i < iNumElements; i += iNumThreads)
s_memoryA[i] = r_var;
/* Conditionally assign register var, so it won't get optimized */
if(iID == 0) outFloat[0] = r_var;
}