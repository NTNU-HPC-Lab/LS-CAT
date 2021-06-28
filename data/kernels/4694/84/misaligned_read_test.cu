#include "includes.h"
__global__ void misaligned_read_test(float* a, float* b, float *c, int size, int offset)
{
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int k = gid + offset;

if (k < size)
c[gid] = a[k]+ b[k];

//c[gid] = a[gid];
}