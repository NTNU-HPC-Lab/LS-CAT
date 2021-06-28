#include "includes.h"
__device__ float3 color(unsigned int depth, unsigned int maxDepth)
{
if(depth == maxDepth)
return make_float3(0.0f, 0.0f, 0.0f);
else
return make_float3(1.0f, 1.0f, 1.0f);
}
__device__ unsigned int mandelDepth(float cr, float ci, int maxDepth)
{
float zr = 0.0f;
float zi = 0.0f;
float zrSqr = 0.0f;
float ziSqr = 0.0f;

unsigned int i;

for (i = 0; i < maxDepth; i++)
{
zi = zr * zi;
zi += zi + ci;
zr = zrSqr - ziSqr + cr;
zrSqr = zr * zr;
ziSqr = zi * zi;

if (zrSqr + ziSqr > 4.0f) break;
}

return i;
}
__global__ void mandel(float* buffer, float xMin, float xMax, float yMin, float yMax, unsigned int maxDepth)
{
int nx = blockDim.x;
int ny = gridDim.x;
float dx = (xMax - xMin) / nx;
float dy = (yMax - yMin) / ny;
float x = xMin + (threadIdx.x + 0.5f) * dx;
float y = yMin + (blockIdx.x + 0.5f) * dy;

unsigned int depth = mandelDepth(x, y, maxDepth);
float3 depthColor = color(depth, maxDepth);

int index = blockIdx.x*blockDim.x + threadIdx.x;
buffer[3*index + 0] = depthColor.x;
buffer[3*index + 1] = depthColor.y;
buffer[3*index + 2] = depthColor.z;
}