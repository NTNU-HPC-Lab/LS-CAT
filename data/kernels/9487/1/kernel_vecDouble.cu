#include "includes.h"
__global__ void kernel_vecDouble(int *in, int *out, const int n)
{
int i = threadIdx.x;
if (i < n) {
out[i] = in[i] * 2;
}
}