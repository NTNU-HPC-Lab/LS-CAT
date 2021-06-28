#include "includes.h"
__global__ void updatePosition_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{
int elementId = blockIdx.x * blockDim.x + threadIdx.x;

float4 elementPosMass;
float3 elementSpeed;

if (elementId < numElements) {
elementPosMass = bodyPos[elementId];
elementSpeed = bodySpeed[elementId];

elementPosMass.x += elementSpeed.x * TIMESTEP;
elementPosMass.y += elementSpeed.y * TIMESTEP;
elementPosMass.z += elementSpeed.z * TIMESTEP;

bodyPos[elementId] = elementPosMass;
}
}