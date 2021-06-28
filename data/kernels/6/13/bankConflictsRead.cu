#include "includes.h"
__global__ void bankConflictsRead(float *outFloat, int iStride, unsigned long long *ullTime)
{
/* Static size of shared memory */
__shared__ float s_memoryA[2024];
/* Variable in register */
float r_var;
/* Start measure clock cycles */
unsigned long long startTime = clock64();
/* Access data from shared memory to register */
r_var = s_memoryA[threadIdx.x*iStride];
/* End measure clock cycles */
*ullTime = clock64() - startTime;
/* Conditionally assign register var, so it won't get optimized */
if(threadIdx.x == 0) outFloat[0] = r_var;
}