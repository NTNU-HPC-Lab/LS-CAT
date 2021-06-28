#include "includes.h"
__global__ void integrateBins(int width, int height, int nbins, int* devImage, int binPitch, int* devIntegrals) {
__shared__ int pixels[16];
const int blockX = blockDim.y * blockIdx.x;
const int threadX = threadIdx.y;
const int bin = threadIdx.x;
const int x = blockX + threadX;
if (x >= width) return;
if (bin > nbins) return;
int* imagePointer = devImage + x;
int* outputPointer = devIntegrals + binPitch * x + bin;
int accumulant = 0;
for(int y = 0; y < height; y++) {
if (bin == 0) {
pixels[threadX] = *imagePointer;
}
__syncthreads();
if (pixels[threadX] == bin) accumulant++;
*outputPointer = accumulant;
imagePointer += width;
outputPointer += width * binPitch;
}
}