#include "includes.h"
__global__ void kernel_update( float4* d_positions, float4* d_og_positions, float4* d_velocities, float* d_masses, size_t numel) {

size_t col = threadIdx.x + blockIdx.x * blockDim.x;
if (col >= numel) { return; }

float4 velocity = d_velocities[col];

float mag = sqrtf(velocity.x*velocity.x + velocity.y*velocity.y)*0.03;
float pos = min(mag, 0.50f);
d_positions[col] = make_float4(
d_og_positions[col].x,
d_og_positions[col].y,
pos, 0
);
__syncthreads();
}