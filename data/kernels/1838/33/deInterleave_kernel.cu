#include "includes.h"
__global__ void deInterleave_kernel(float *d_X_out, float *d_Y_out, float2 *d_XY_in, int pitch_out, int pitch_in, int width, int height) {
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < width) & (y < height)) { // are we in the image?
float2 XY = *((float2 *)((char *)d_XY_in + y * pitch_in) + x);
*((float *)((char *)d_X_out + y *pitch_out) + x) = XY.x;
*((float *)((char *)d_Y_out + y *pitch_out) + x) = XY.y;
}
}