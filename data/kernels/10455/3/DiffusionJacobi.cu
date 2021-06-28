#include "includes.h"
__device__ float2 JacobiFieldInstance(float2 Top, float2 Left, float2 Bot, float2 Right, float Alpha, float2 Val)
{
float2 res;
res.x = (Top.x + Left.x + Bot.x + Right.x + Alpha * Val.x) / (4 + Alpha);
res.y = (Top.y + Left.y + Bot.y + Right.y + Alpha * Val.y) / (4 + Alpha);
return res;
}
__global__ void DiffusionJacobi(float2 *__restrict__ positions, float2 *__restrict__ oldVel, float2 *__restrict__ newVel, float dt, float dr, float viscosity, unsigned int simWidth, unsigned int simHeight)
{
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

float2 Vel = oldVel[y*simWidth + x];
float2 TVel;
float2 LVel;
float2 BVel;
float2 RVel;
float alpha = dr * dr / (viscosity * dt);

if (x!=0 && y!=0 && x!=simWidth-1 && y!=simHeight-1)
{

TVel = oldVel[(y-1)*simWidth + x];
LVel = oldVel[(y*simWidth) + x - 1];
BVel = oldVel[(y+1)*simWidth + x];
RVel = oldVel[(y*simWidth) + x + 1];

newVel[y*simWidth + x] = JacobiFieldInstance(TVel, LVel,
BVel, RVel,
alpha, Vel);
}
}