#include "includes.h"
__global__ void sobelEdgeDetectionSharedMemOverlap(int *input, int *output, int width, int height, int thresh) {

static __shared__ int shMem[_TILESIZE_2 * _TILESIZE_2];

int blocksize = _TILESIZE_2;
int i = blockIdx.x * (_TILESIZE_) + threadIdx.x;
int j = blockIdx.y * (_TILESIZE_) + threadIdx.y;
int index = j * width + i;

int xind = threadIdx.x;
int yind = threadIdx.y;

shMem[blocksize * yind + xind] = input[index];
__syncthreads();

if ( xind > 0 && yind > 0 && xind < (blocksize - 1) && yind < (blocksize - 1))
{

int sum1 = shMem[xind + 1 + blocksize * (yind - 1)] -     shMem[xind - 1 + blocksize * (yind - 1)]
+ 2 * shMem[xind + 1 + blocksize * (yind    )] - 2 * shMem[xind - 1 + blocksize * (yind    )]
+     shMem[xind + 1 + blocksize * (yind + 1)] -     shMem[xind - 1 + blocksize * (yind + 1)];

int sum2 = shMem[xind - 1 + blocksize * (yind - 1)] + 2 * shMem[xind     + blocksize * (yind - 1)] + shMem[xind + 1 + blocksize * (yind - 1)]
- shMem[xind - 1 + blocksize * (yind + 1)] - 2 * shMem[xind     + blocksize * (yind + 1)] - shMem[xind + 1 + blocksize * (yind + 1)];

int magnitude = sum1 * sum1 + sum2 * sum2;
if(magnitude > thresh)
output[index] = 255;
else
output[index] = 0;
}
}