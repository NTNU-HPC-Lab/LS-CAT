#include "includes.h"
__global__ void simple_input_shortcut_kernel(float *in, int size, float *add, float *out)
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= size) return;

out[id] = in[id] + add[id];
}