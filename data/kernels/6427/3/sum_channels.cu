#include "includes.h"



__global__ static void sum_channels(float *dest, const float *src, uint channels, uint num_channel_elem)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx >= num_channel_elem)
return;

float acc = 0;
for (uint i = 0; i < channels; ++i)
acc += src[idx + i * num_channel_elem];
dest[idx] = acc;
}