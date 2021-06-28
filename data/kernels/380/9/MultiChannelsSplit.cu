#include "includes.h"
__global__ void MultiChannelsSplit(float* inputs, float* outputs, int outChannels, int offset, int row, int inChannels)
{
int  batchId = blockIdx.x;
float* input = inputs + batchId * inChannels * row * row + offset;

int blockDo  = outChannels * row * row;
for(int i = 0; i < blockDo; i += blockDim.x)
{
int j = i + threadIdx.x;
if(j < blockDo)
{
int pos = batchId * outChannels * row * row;
outputs[pos + j] = input[j];
}
}
}