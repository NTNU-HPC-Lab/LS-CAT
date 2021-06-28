#include "includes.h"
__global__ void optimalTransposeKernel(const float *input, float *output, int n)
{
__shared__ float tile[64][65];
int x = blockIdx.x * 64 + threadIdx.x;
int y = blockIdx.y * 64 + threadIdx.y;
const int width = gridDim.x * 64;
const int height = gridDim.y * 64;
if (x < width && y < height)
{ tile[threadIdx.y][threadIdx.x] = input[y*width + x];
tile[threadIdx.y+16][threadIdx.x] = input[(y+16)*width +x];
tile[threadIdx.y+32][threadIdx.x] = input[(y+32)*width +x];
tile[threadIdx.y+48][threadIdx.x] = input[(y+48)*width +x];
}
__syncthreads();

x = blockIdx.y * 64 + threadIdx.x; // transpose block offset
y = blockIdx.x * 64 + threadIdx.y;
if (y < width && x < height)
{ output[y*height + x] = tile[threadIdx.x][threadIdx.y];
output[(y+16)*height +x] = tile[threadIdx.x][threadIdx.y+16];
output[(y+32)*height +x] = tile[threadIdx.x][threadIdx.y+32];
output[(y+48)*height +x] = tile[threadIdx.x][threadIdx.y+48];
}
}