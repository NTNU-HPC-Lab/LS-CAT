#include "includes.h"
__global__ void CopyChannel_i_Kernel(float* output, const float* input, const int i, const int width, const int height, const int nChannels)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x >= width || y >= height)
return;
int offset = y*width + x;
output[offset] = input[offset*nChannels + i];
}