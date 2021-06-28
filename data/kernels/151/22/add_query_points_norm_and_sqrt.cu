#include "includes.h"
__global__ void add_query_points_norm_and_sqrt(float * array, int width, int pitch, int k, float * norm){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
if (xIndex<width && yIndex<k)
array[yIndex*pitch + xIndex] = sqrt(array[yIndex*pitch + xIndex] + norm[xIndex]);
}