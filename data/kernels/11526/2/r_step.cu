#include "includes.h"
__global__ void r_step( float4 *__restrict__ devPos, float4 *__restrict__ deviceVel, unsigned int numBodies, float dt)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index > numBodies) {return;};
devPos[index].x += deviceVel[index].x * dt;
devPos[index].y += deviceVel[index].y * dt;
devPos[index].z += deviceVel[index].z * dt;
}