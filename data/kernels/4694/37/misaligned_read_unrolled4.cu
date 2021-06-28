#include "includes.h"
__global__ void misaligned_read_unrolled4(int* a, int* b, int *c, int size, int offset)
{
int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
int k = i + offset;

if (k + 3 * blockDim.x < size)
{
c[i] = a[k] + b[k];
c[i + blockDim.x] = a[k + blockDim.x] + b[k + blockDim.x];
c[i + 2* blockDim.x] = a[k + 2 * blockDim.x] + b[k + 2 *blockDim.x];
c[i + 3* blockDim.x] = a[k + 3* blockDim.x] + b[k + 3* blockDim.x];
}
}