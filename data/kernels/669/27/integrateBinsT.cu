#include "includes.h"
__global__ void integrateBinsT(int width, int height, int nbins, int binPitch, int* devIntegrals) {
const int blockY = blockDim.y * blockIdx.x;
const int threadY = threadIdx.y;
const int bin = threadIdx.x;
const int y = blockY + threadY;
if (y >= height) return;
if (bin >= binPitch) return;
int* imagePointer = devIntegrals + binPitch * y * width + bin;
int accumulant = 0;
for(int x = 0; x < width; x++) {
accumulant += *imagePointer;
*imagePointer = accumulant;
imagePointer += binPitch;
}
}