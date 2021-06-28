#include "includes.h"
__global__ void setToZeros(float *data, int size)
{
int index = threadIdx.x + blockIdx.x * blockDim.x; // 1D grid of 1D blocks
if (index < size) data[index] = 0;
}