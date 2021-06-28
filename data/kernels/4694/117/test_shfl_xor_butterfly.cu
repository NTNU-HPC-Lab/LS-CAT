#include "includes.h"
__global__ void test_shfl_xor_butterfly(int * in, int *out)
{
int x = in[threadIdx.x];
int y = __shfl_xor_sync(0xffffffff, x, 1, 32);
out[threadIdx.x] = y;
}