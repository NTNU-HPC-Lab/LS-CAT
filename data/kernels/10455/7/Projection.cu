#include "includes.h"
__global__ void Projection(float2 *__restrict__ newVel, float2 *__restrict__ gradPressure, unsigned int simWidth)
{
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

newVel[y*simWidth+x].x -= gradPressure[y*simWidth+x].x;
newVel[y*simWidth+x].y -= gradPressure[y*simWidth+x].y;
}