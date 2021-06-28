#include "includes.h"



#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)




__global__ void erodeDepthMapDevice(float* d_output, float* d_input, int structureSize, int width, int height, float dThresh, float fracReq)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;


if (x >= 0 && x < width && y >= 0 && y < height)
{


unsigned int count = 0;

float oldDepth = d_input[y*width + x];
for (int i = -structureSize; i <= structureSize; i++)
{
for (int j = -structureSize; j <= structureSize; j++)
{
if (x + j >= 0 && x + j < width && y + i >= 0 && y + i < height)
{
float depth = d_input[(y + i)*width + (x + j)];
if (depth == MINF || depth == 0.0f || fabs(depth - oldDepth) > dThresh)
{
count++;
//d_output[y*width+x] = MINF;
//return;
}
}
}
}

unsigned int sum = (2 * structureSize + 1)*(2 * structureSize + 1);
if ((float)count / (float)sum >= fracReq) {
d_output[y*width + x] = MINF;
}
else {
d_output[y*width + x] = d_input[y*width + x];
}
}
}