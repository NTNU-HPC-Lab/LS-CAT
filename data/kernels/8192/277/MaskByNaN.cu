#include "includes.h"
__global__ void MaskByNaN( float* inputImage, float* mask, float* outputImage, int count ) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;

if (id < count)
{
if (mask[id] == 0.0f)
{
outputImage[id] = NAN;
}
else {
outputImage[id] = inputImage[id];
}
}
}