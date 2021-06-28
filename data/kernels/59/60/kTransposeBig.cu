#include "includes.h"
__global__ void kTransposeBig(float *odata, float *idata, int height, int width) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
int r, c;
for (unsigned int i = idx; i < width * height; i += numThreads) {
r = i % width;
c = i / width;
odata[i] = idata[height * r + c];
}
}