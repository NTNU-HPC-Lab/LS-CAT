#include "includes.h"
///////////////////////////////////////////////////////////////////////////////

//Round a / b to nearest higher integer value
__global__ void calculateSlopeKernel(float* h, float2 *slopeOut, unsigned int width, unsigned int height){
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
unsigned int i = y*width+x;

float2 slope = make_float2(0.0f, 0.0f);
if ((x > 0) && (y > 0) && (x < width-1) && (y < height-1)) {
slope.x = h[i+1] - h[i-1];
slope.y = h[i+width] - h[i-width];
}
slopeOut[i] = slope;
}