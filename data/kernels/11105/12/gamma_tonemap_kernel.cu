#include "includes.h"
__device__ float gamma_correction(float f_stop, float gamma, float val)
{
return powf((val*powf(2.0,f_stop)),(1.0/gamma));
}
__device__ float rgb2Lum(float B, float G, float R)
{
return B * 0.0722 + G * 0.7152 + R * 0.2126;
}
__global__ void gamma_tonemap_kernel(float* imageIn, float* imageOut, int width, int height, int channels, float f_stop, float gamma)
{
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;

if(Row < height && Col < width) {
float B, G, R, L, nL, scale;
B = imageIn[(Row*width+Col)*3+BLUE];
G = imageIn[(Row*width+Col)*3+GREEN];
R = imageIn[(Row*width+Col)*3+RED];

L = rgb2Lum(B, G, R);
nL = gamma_correction(f_stop, gamma, L);
scale = nL / L;

imageOut[(Row*width+Col)*3+BLUE] = B * scale;
imageOut[(Row*width+Col)*3+GREEN] = G * scale;
imageOut[(Row*width+Col)*3+RED] = R * scale;
}
}