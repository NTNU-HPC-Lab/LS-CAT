#include "includes.h"
__global__ void vectorAdd(const double *A, const double *B, double *C, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElements)
{
C[i] = A[i] + B[i];
}
printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
"gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
gridDim.x,gridDim.y,gridDim.z);
}