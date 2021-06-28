#include "includes.h"
__global__ void kernel_grey( float4* d_Iin, float* d_Iout, int numel ) {

size_t col = threadIdx.x + blockDim.x * blockIdx.x;
if (col >= numel) {
return;
}

float4 pixel = d_Iin[col];

d_Iout[col] = 0.2989f * (pixel.x) + 0.5870f * (pixel.y) + 0.1140f * (pixel.z);
}