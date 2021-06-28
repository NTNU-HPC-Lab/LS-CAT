#include "includes.h"
__global__ void add_reference_points_norm(float * array, int width, int pitch, int height, float * norm){
unsigned int tx = threadIdx.x;
unsigned int ty = threadIdx.y;
unsigned int xIndex = blockIdx.x * blockDim.x + tx;
unsigned int yIndex = blockIdx.y * blockDim.y + ty;
__shared__ float shared_vec[16];
if (tx==0 && yIndex<height)
shared_vec[ty] = norm[yIndex];
__syncthreads();
if (xIndex<width && yIndex<height)
array[yIndex*pitch+xIndex] += shared_vec[ty];
}