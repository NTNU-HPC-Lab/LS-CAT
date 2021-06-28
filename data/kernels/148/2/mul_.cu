#include "includes.h"


#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

__global__ void mul_(float *input, float factor, int size)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
input[id] = input[id] * factor;
}
}