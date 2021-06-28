#include "includes.h"
__global__ void cuAddRNorm(float *dist, int width, int pitch, int height, float *vec){
unsigned int tx = threadIdx.x;
unsigned int ty = threadIdx.y;
unsigned int xIndex = blockIdx.x * blockDim.x + tx;
unsigned int yIndex = blockIdx.y * blockDim.y + ty;
__shared__ float shared_vec[16];
if (tx==0 && yIndex<height)
shared_vec[ty]=vec[yIndex];
__syncthreads();
if (xIndex<width && yIndex<height)
dist[yIndex*pitch+xIndex]+=shared_vec[ty];
}