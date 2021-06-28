#include "includes.h"
__global__ void backward_dropblock_kernel(float *pass, float *delta, int size)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= size) return;

if (pass[index] == 0) delta[index] = 0;
}