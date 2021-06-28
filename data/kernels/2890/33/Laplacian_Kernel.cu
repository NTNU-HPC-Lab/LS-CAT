#include "includes.h"
__global__ void Laplacian_Kernel(float* output, const float* input, const int width, const int height, const int nChannels)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x >= width || y >= height)
return;

int offset = y*width + x;

for (int c = 0; c < nChannels; c++)
{
float value = 0;
if (x == 0)
{
value += input[(offset + 1)*nChannels + c] - input[offset*nChannels + c];
}
else if (x == width - 1)
{
value += input[(offset - 1)*nChannels + c] - input[offset*nChannels + c];
}
else
{
value += input[(offset + 1)*nChannels + c] + input[(offset - 1)*nChannels + c] - 2 * input[offset*nChannels + c];
}

if (y == 0)
{
value += input[(offset + width)*nChannels + c] - input[offset*nChannels + c];
}
else if (y == height - 1)
{
value += input[(offset - width)*nChannels + c] - input[offset*nChannels + c];
}
else
{
value += input[(offset + width)*nChannels + c] + input[(offset - width)*nChannels + c] - 2 * input[offset*nChannels + c];
}

output[offset*nChannels + c] = value;
}
}