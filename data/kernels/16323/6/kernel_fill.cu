#include "includes.h"
__global__ void kernel_fill(float4* d_dx1, float val, int numel) {
size_t col = threadIdx.x + blockIdx.x * blockDim.x;
if (col >= numel) { return; }

d_dx1[col].x = val;
d_dx1[col].y = val;
d_dx1[col].z = val;
d_dx1[col].w = val;
}