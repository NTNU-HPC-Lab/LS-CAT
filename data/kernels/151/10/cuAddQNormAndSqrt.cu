#include "includes.h"
__global__ void cuAddQNormAndSqrt(float *vec1,  float *vec2, int width){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
if (xIndex<width){
vec1[xIndex] = sqrt(vec1[xIndex]+vec2[xIndex]);
}
}