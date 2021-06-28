#include "includes.h"
__global__ void test_shfl_up(int * in, int *out)
{
int x = in[threadIdx.x];
int y = __shfl_up_sync(0xffffffff, x, 3);
out[threadIdx.x] = y;
}