#include "includes.h"
__global__ void cuAddQNormAndSqrt(float *dist, int width, int pitch, float *q, int k){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
if (xIndex<width && yIndex<k)
dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex] + q[xIndex]);
}