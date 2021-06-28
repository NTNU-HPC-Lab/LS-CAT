#include "includes.h"
__global__ void Add_Im1_weight1_Im2_weight2_Kernel(float* output, const float* Im1, const float weight1, const float* Im2, const float weight2, const int width, const int height, const int nChannels)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x >= width || y >= height)
return;

int offset = y*width + x;
for (int c = 0; c < nChannels; c++)
{
output[offset*nChannels + c] = Im1[offset*nChannels + c] * weight1 + Im2[offset*nChannels + c] * weight2;
}
}