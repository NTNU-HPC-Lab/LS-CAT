#include "includes.h"
__device__ void EncodeValuesInternal(float value, float& origin, float& dir, float& output, int squaredMode)
{
if (squaredMode == 1)
{
// origin part:      o * (1 - t)^2
output = (1 - fabs(value)) * (1 - fabs(value)) * origin;
// direction part:   dir * (-t^2 + 2*t)
output += (-value * value + 2 * fabs(value)) * dir;
}
else
{
// origin part:      o * (1 - t)
output = (1 - fabs(value)) * origin;
// direction part:   dir * t
output += fabs(value) * dir;
}
}
__global__  void EncodeValues(float* values, int numOfValues, float* output, int symbolSize, int squaredMode, float* dirX, float* dirY, float* negDirX, float* negDirY, float* originX, float* originY)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

__shared__ float s_values[2];

if (threadIdx.x < 2)
{
//clamp to (-1, 1) if square mode is used

if (squaredMode == 1)
{
s_values[threadIdx.x] = fmaxf(fminf(values[threadIdx.x], 1), -1);
}
else
{
s_values[threadIdx.x] = values[threadIdx.x];
}
}

__syncthreads();


if (threadId >= symbolSize)
return;


// X dim
float* dir = (s_values[0] > 0) ? dirX : negDirX;
EncodeValuesInternal(s_values[0], originX[threadId], dir[threadId], output[threadId], squaredMode);

// Y dim
if (numOfValues > 1)
{
dir = (s_values[1] > 0) ? dirY : negDirY;
EncodeValuesInternal(s_values[1], originY[threadId], dir[threadId], output[threadId], squaredMode);
}
}