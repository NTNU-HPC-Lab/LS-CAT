#include "includes.h"
__global__ void kernel(unsigned char *ptr, int ticks)
{
//Index one of the threads to an image pos
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

float fx = x - DIM / 2;
float fy = y - DIM / 2;
float d = sqrtf(fx * fx + fy * fy);
//Create varying grey vals depending on pixel
unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

//Offset into output buffer for window generation when ready
ptr[offset * 4 + 0] = grey;
ptr[offset * 4 + 1] = grey;
ptr[offset * 4 + 2] = grey;
ptr[offset * 4 + 3] = 255;
}