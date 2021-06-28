#include "includes.h"


#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

__global__ void fill_(float *input, float value, int size)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
input[id] = value;
}
}