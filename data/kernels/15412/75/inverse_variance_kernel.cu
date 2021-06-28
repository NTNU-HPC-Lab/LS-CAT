#include "includes.h"
__global__ void inverse_variance_kernel(int size, float *src, float *dst, float epsilon)
{
int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < size)
dst[index] = 1.0f / sqrtf(src[index] + epsilon);
}