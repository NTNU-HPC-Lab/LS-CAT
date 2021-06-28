#include "includes.h"
__global__ void Dx_Forward_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x >= width || y >= height)
return;

int offset = y*width + x;
if (x == width - 1)
{
for (int c = 0; c < nChannels; c++)
output[offset*nChannels + c] = 0;
}
else
{
for (int c = 0; c < nChannels; c++)
output[offset*nChannels + c] = input[(offset + 1)*nChannels + c] - input[offset*nChannels + c];
}
}