#include "includes.h"
__global__ void kernel_update_models(float4* d_positions, float4* d_modelBuffer, int numel) {

size_t col = threadIdx.x + blockIdx.x * blockDim.x;
if (col >= numel) { return; }

d_modelBuffer[col*4+3] = make_float4(
d_positions[col].x,
d_positions[col].y,
d_positions[col].z,
1
);
__syncthreads();
}