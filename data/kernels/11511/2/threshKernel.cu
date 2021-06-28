#include "includes.h"
__global__ void threshKernel(unsigned char * image, unsigned char* moddedimage, int size, int threshold)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size)
{
if (image[i] > threshold)
{
moddedimage[i] = 255;
}
else
{
moddedimage[i] = 0;
}
}
}