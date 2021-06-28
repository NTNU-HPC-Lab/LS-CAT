#include "includes.h"
__global__ void vel_step( float4 *__restrict__ deviceVel, float3 *__restrict__ accels, unsigned int numBodies, float dt)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index > numBodies) {return;};
deviceVel[index].x += accels[index].x * 0.5 * dt;
deviceVel[index].y += accels[index].y * 0.5 * dt;
deviceVel[index].z += accels[index].z * 0.5 * dt;
}