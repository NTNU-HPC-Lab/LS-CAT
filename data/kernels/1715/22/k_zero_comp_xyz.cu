#include "includes.h"
__global__ static void k_zero_comp_xyz(float *data, uint n, uint stride)
{
uint i = blockIdx.x * blockDim.x + threadIdx.x;
uint p = blockIdx.y;

if (i < n) {
data[i + p * stride] = 0.f;
}
}