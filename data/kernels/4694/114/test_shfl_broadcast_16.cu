#include "includes.h"
__global__ void test_shfl_broadcast_16(int * in, int *out)
{
int x = in[threadIdx.x];
int y = __shfl_sync(0xffffffff, x, 3, 16);
out[threadIdx.x] = y;
}