#include "includes.h"



#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ );

__global__ void render( float* framebuffer, int width, int height )
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

if( i >= width || j >= height )
{
return;
}

int pixel_index = j * width * 3 + i * 3;

framebuffer[pixel_index + 0] = float(i) / width;
framebuffer[pixel_index + 1] = float(j) / height;
framebuffer[pixel_index + 2] = 0.2f;
}