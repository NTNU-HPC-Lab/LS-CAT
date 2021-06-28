#include "includes.h"
__global__ void iota(int const size, int *data, int const value)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < size)
data[idx] = idx + value;
}