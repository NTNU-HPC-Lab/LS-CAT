#include "includes.h"
__global__ void kernelCalculateHistogram(unsigned int* histogram, unsigned char* rawPixels, long chunkSize, long totalPixels)
{
int id = blockDim.x * blockIdx.x + threadIdx.x;

int startPosition = id * chunkSize;
for (int i = startPosition; i < (startPosition + chunkSize); i++) {
if (i < totalPixels) {
int pixelValue = (int)rawPixels[i];
atomicAdd(&histogram[pixelValue], 1);
}
}
}