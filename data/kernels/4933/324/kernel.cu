#include "includes.h"
__global__ void kernel(float *g_data, float value)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
g_data[idx] = g_data[idx] + value;
}