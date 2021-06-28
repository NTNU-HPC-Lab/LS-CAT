#include "includes.h"


#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

__global__ void downsample_(float *input, float *output, int factor, int size3, int size)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size) {
int dim3 = id % size3;
int dim2 = id / size3;
atomicAdd(output + ((dim2 / factor) * (size3 / factor) + (dim3 / factor)), input[id] / (factor * factor));
}
}