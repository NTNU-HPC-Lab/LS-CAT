#include "includes.h"
__global__ void deInterleave_kernel2(float *d_X_out, float *d_Y_out, char *d_XY_in, int pitch_out, int pitch_in, int width, int height) {
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

if ((x < width) & (y < height)) { // are we in the image?
float *data = (float *)(d_XY_in + y * pitch_in) + 2 * x;
*((float *)((char *)d_X_out + y *pitch_out) + x) = data[0];
*((float *)((char *)d_Y_out + y *pitch_out) + x) = data[1];
}
}