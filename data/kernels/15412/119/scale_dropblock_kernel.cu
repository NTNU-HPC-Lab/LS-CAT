#include "includes.h"
__global__ void scale_dropblock_kernel(float *output, int size, int outputs, float *drop_blocks_scale)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= size) return;

const int b = index / outputs;
output[index] *= drop_blocks_scale[b];
}