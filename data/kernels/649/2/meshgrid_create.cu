#include "includes.h"
__global__ void meshgrid_create(float* xx, float* yy, int w, int h, float K02, float K12) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
if (i < h && j < w) {
xx[j*h + i] = j - K02;
yy[j*h + i] = i - K12;
}
}