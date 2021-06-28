#include "includes.h"
__global__ void updateVel(float2 *__restrict__ oldVel, float2 *__restrict__ newVel, unsigned int simWidth)
{
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
oldVel[y*simWidth+x] = newVel[y*simWidth+x];
}