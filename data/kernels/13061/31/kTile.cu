#include "includes.h"
__global__ void kTile(float* src, float* tgt, unsigned int srcWidth, unsigned int srcHeight, unsigned int tgtWidth, unsigned int tgtHeight) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
//    const unsigned int numEls = tgtWidth * tgtHeight;
for (unsigned int i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
const unsigned int y = i / tgtWidth;
const unsigned int x = i % tgtWidth;
const unsigned int srcY = y % srcHeight;
const unsigned int srcX = x % srcWidth;
tgt[i] = src[srcY * srcWidth + srcX];
}
}