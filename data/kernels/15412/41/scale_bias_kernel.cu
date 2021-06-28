#include "includes.h"
__global__ void scale_bias_kernel(float *output, float *scale, int batch, int filters, int spatial, int current_size)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= current_size) return;

int f = (index / spatial) % filters;
output[index] *= scale[f];
}