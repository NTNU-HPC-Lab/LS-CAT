#include "includes.h"
__global__ void histogramm(float* hist, unsigned char* input, int width, int height, int stride)
{
int index = blockIdx.x * blockDim.x * stride + threadIdx.x;
int size = width * height;
if (index > size - 1)
return;

__shared__ unsigned int histo_private[256];

#pragma unroll
for (int i = 0; i < 8; i++)
{
histo_private[threadIdx.x * 8 + i] = 0;
}

__syncthreads();

int i = 0;
while (i < stride && index < size)
{
int pixel = input[index];
atomicAdd(&(histo_private[pixel]), 1);
index += blockDim.x;
i++;
}

__syncthreads();

#pragma unroll
for (int i = 0; i < 8; i++)
{
int x_off = threadIdx.x * 8 + i;
hist[x_off * 3 + 0] = (x_off - 128.f) / 256.f * (float)width;

float factor = .48f;
float scaledValue = ((float)(histo_private[x_off]) / (float)size) - (factor / gridDim.x);
atomicAdd(&(hist[x_off * 3 + 1]), scaledValue * (float)height);
}
}