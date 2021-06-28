#include "includes.h"
__global__ void convertKinectDisparityInPlace_kernel(float *d_disparity, int pitch, int width, int height, float depth_scale) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < width) & (y < height)) { // are we in the image?

float *d_in = (float *)((char *)d_disparity + y * pitch) + x;
*d_in = (*d_in == 0.0f) ? nanf("") : (-depth_scale / *d_in);
}
}