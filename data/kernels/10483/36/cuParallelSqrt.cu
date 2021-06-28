#include "includes.h"
__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
if (xIndex<width && yIndex<k)
dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}