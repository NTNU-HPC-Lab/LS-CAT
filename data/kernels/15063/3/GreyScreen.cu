#include "includes.h"
__global__ void GreyScreen(float* d_pixelsR, float* d_pixelsG, float* d_pixelsB, float* d_reducePixels, int numPixels){
int id = threadIdx.x + blockIdx.x * blockDim.x;
//printf("Test ID: %u ", numPixels);
if (id < numPixels){
d_reducePixels[id] = (d_pixelsR[id] + d_pixelsG[id] + d_pixelsB[id]) / 3;
//printf("Reduce Pixels ");
//printf("%f ", d_reducePixels[id]);
}
}