#include "includes.h"
__global__ void PrepareMeanStdDev(float* input, float* delta, int imageWidth, int imageHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int size = imageWidth * imageHeight;

if (id < size)
{
int px = id % imageWidth;
int py = id / imageWidth;

float2 pixPos = {  2.0f * px / imageWidth - 1,  2.0f * py / imageHeight - 1};

//mean sum
delta[id] = input[id] * pixPos.x;
delta[id + size] = input[id] * pixPos.y;

//variance sum
delta[id + 2 * size] = input[id] * pixPos.x * pixPos.x;
delta[id + 3 * size] = input[id] * pixPos.y * pixPos.y;
}
}