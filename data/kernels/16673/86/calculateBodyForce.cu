#include "includes.h"
__global__ void calculateBodyForce(float4 *p, float4 *v, float dt, int n) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < n) {
float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

for (int tile = 0; tile < gridDim.x; tile++) {
__shared__ float3 shared_position[BLOCK_SIZE];
float4 temp_position = p[tile * blockDim.x + threadIdx.x];
shared_position[threadIdx.x] = make_float3(temp_position.x, temp_position.y, temp_position.z);
__syncthreads(); //synchronoze to make sure all tile data is available in shared memory

for (int j = 0; j < BLOCK_SIZE; j++) {
float dx = shared_position[j].x - p[i].x;
float dy = shared_position[j].y - p[i].y;
float dz = shared_position[j].z - p[i].z;
float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
float invDist = rsqrtf(distSqr);
float invDist3 = invDist * invDist * invDist;

Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
}
__syncthreads(); // synchrnize before looping to other time
} //tile loop ends here

v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
} //if ends here
}