#include "includes.h"
__device__ inline int getTransArrayIndex(unsigned int width, unsigned int height, unsigned  int i) {
return height * (i % width) + i / width;
}
__global__ void kAddTransSlow(float* a, float* b, float* dest, unsigned int width, unsigned int height, unsigned int numEls, float scaleA, float scaleB) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
//    const unsigned int idx = blockIdx.y * height + blockIdx.x * blockDim.x  + threadIdx.y*blockDim.x + threadIdx.x;
for (unsigned int i = idx; i < numEls; i += numThreads) {
dest[i] = scaleA * a[i] + scaleB * b[getTransArrayIndex(width, height, i)];
}
}