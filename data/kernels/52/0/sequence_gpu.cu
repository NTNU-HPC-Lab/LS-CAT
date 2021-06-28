#include "includes.h"
__global__ void sequence_gpu(int *d_ptr, int length)
{
int elemID = blockIdx.x * blockDim.x + threadIdx.x;

if (elemID < length)
{
d_ptr[elemID] = elemID;
}
}