#include "includes.h"



__global__ void kernel_update_velocities(float4* d_uv, float4* d_velocities_buffer, int numel) {

size_t col = threadIdx.x + blockIdx.x * blockDim.x;
if (col >= numel) { return; }

d_velocities_buffer[col] = make_float4(
d_uv[col].x,
d_uv[col].y,
0,
0
);
__syncthreads();
}