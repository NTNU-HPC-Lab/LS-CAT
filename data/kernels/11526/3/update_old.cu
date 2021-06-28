#include "includes.h"
__global__ void update_old( float4 *__restrict__ newPos, float4 *__restrict__ oldPos )
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
oldPos[index] = newPos[index];
}