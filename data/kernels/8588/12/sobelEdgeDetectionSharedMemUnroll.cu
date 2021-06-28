#include "includes.h"
__global__ void sobelEdgeDetectionSharedMemUnroll(int *input, int *output, int width, int height, int thresh) {

__shared__ int shMem[4 * _TILESIZE_2 * _TILESIZE_2 ];

int num = _UNROLL_;
int size = num * _TILESIZE_2;

int i = blockIdx.x * num * _TILESIZE_ + threadIdx.x * num;
int j = blockIdx.y * num * _TILESIZE_ + threadIdx.y * num;

int xind = num * threadIdx.x;
int yind = num * threadIdx.y;

for(int x = 0; x < num; x++)
{
for(int y = 0; y < num; y++)
{
shMem[ size * (yind + y) + (xind + x)] = input[(j + y) * width + (i + x)];
}
}

__syncthreads();

if ( xind > 0 && yind > 0 && xind < (size - 2) && yind < (size - 2))
{
for(int x = 0; x < num; x++)
{
for(int y = 0; y < num; y++)
{

int sum1 = shMem[(xind + 1 + x) + size * (yind - 1 + y)] -     shMem[(xind - 1 + x) + size * (yind - 1 + y)]
+ 2 * shMem[(xind + 1 + x) + size * (yind     + y)] - 2 * shMem[(xind - 1 + x) + size * (yind     + y)]
+     shMem[(xind + 1 + x) + size * (yind + 1 + y)] -     shMem[(xind - 1 + x) + size * (yind + 1 + y)];

int sum2 = shMem[(xind - 1 + x) + size * (yind - 1 + y)] + 2 * shMem[(xind     + x) + size * (yind - 1 + y)] + shMem[(xind + 1 + x) + size * (yind - 1 + y)]
- shMem[(xind - 1 + x) + size * (yind + 1 + y)] - 2 * shMem[(xind     + x) + size * (yind + 1 + y)] - shMem[(xind + 1 + x) + size * (yind + 1 + y)];

int magnitude = sum1 * sum1 + sum2 * sum2;

int index = (j + y) * width + (i + x);

if(magnitude > thresh)
output[index] = 255;
else
output[index] = 0;

}
}
}

}