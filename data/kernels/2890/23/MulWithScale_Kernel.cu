#include "includes.h"
__global__ void MulWithScale_Kernel(float* in_out_put, const float scale, const int width, const int height, const int nChannels)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x >= width || y >= height)
return;

int offset = y*width + x;
for (int c = 0; c < nChannels; c++)
{
in_out_put[offset*nChannels + c] *= scale;
}
}