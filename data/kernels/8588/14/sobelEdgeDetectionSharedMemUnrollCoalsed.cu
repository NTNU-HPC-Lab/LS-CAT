#include "includes.h"
__global__ void sobelEdgeDetectionSharedMemUnrollCoalsed(int *input, int *output, int width, int height, int thresh) {

__shared__ int shMem[4 * _TILESIZE_2 * _TILESIZE_2 ];

int num = _UNROLL_;
int size = num * _TILESIZE_2;

int i = blockIdx.x * (num * _TILESIZE_) + threadIdx.x;
int j = blockIdx.y * (num * _TILESIZE_) + threadIdx.y;

int xind = threadIdx.x;
int yind = threadIdx.y;

for(int x = 0; x < num; x++)
{
for(int y = 0; y < num; y++)
{
int xOffset = x * (_TILESIZE_), yOffset = y * (_TILESIZE_);
shMem[ size * (yind + yOffset) + (xind + xOffset)] = input[(j + yOffset) * width + (i + xOffset)];
}
}

__syncthreads();

if (i < width - _TILESIZE_ && j < height - _TILESIZE_ && xind > 0 && yind > 0 && xind < (_TILESIZE_2 - 1) && yind < (_TILESIZE_2 - 1))
{
for(int x = 0; x < num; x++)
{
for(int y = 0; y < num; y++)
{
int xOffset = x * _TILESIZE_, yOffset = y * _TILESIZE_;

int sum1 = shMem[(xind + 1 + xOffset) + size * (yind - 1 + yOffset)] -     shMem[(xind - 1 + xOffset) + size * (yind - 1 + yOffset)]
+ 2 * shMem[(xind + 1 + xOffset) + size * (yind     + yOffset)] - 2 * shMem[(xind - 1 + xOffset) + size * (yind     + yOffset)]
+     shMem[(xind + 1 + xOffset) + size * (yind + 1 + yOffset)] -     shMem[(xind - 1 + xOffset) + size * (yind + 1 + yOffset)];

int sum2 = shMem[(xind - 1 + xOffset) + size * (yind - 1 + yOffset)] + 2 * shMem[(xind     + xOffset) + size * (yind - 1 + yOffset)] + shMem[(xind + 1 + xOffset) + size * (yind - 1 + yOffset)]
- shMem[(xind - 1 + xOffset) + size * (yind + 1 + yOffset)] - 2 * shMem[(xind     + xOffset) + size * (yind + 1 + yOffset)] - shMem[(xind + 1 + xOffset) + size * (yind + 1 + yOffset)];

int magnitude = sum1 * sum1 + sum2 * sum2;

int index = (j + yOffset) * width + (i + xOffset);

if(magnitude > thresh)
output[index] = 255;
else
output[index] = 0;

}
}
}

}