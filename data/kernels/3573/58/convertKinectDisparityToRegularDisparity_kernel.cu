#include "includes.h"
__global__ void convertKinectDisparityToRegularDisparity_kernel( float *d_regularDisparity, int d_regularDisparityPitch, const float *d_KinectDisparity, int d_KinectDisparityPitch, int width, int height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < width) & (y < height)) { // are we in the image?

float d_in =
*((float *)((char *)d_KinectDisparity + y * d_KinectDisparityPitch) +
x);

float d_out = (d_in == 0.0f) ? nanf("") : -d_in;

*((float *)((char *)d_regularDisparity + y *d_regularDisparityPitch) + x) =
d_out;
}
}