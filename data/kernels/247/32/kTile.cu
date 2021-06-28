#include "includes.h"
__global__ void kTile(const float* src, float* tgt, const int srcWidth, const int srcHeight, const int tgtWidth, const int tgtHeight) {
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
const int numThreads = blockDim.x * gridDim.x;
//    const unsigned int numEls = tgtWidth * tgtHeight;
for (unsigned int i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
const int y = i / tgtWidth;
const int x = i % tgtWidth;
const int srcY = y % srcHeight;
const int srcX = x % srcWidth;
tgt[i] = src[srcY * srcWidth + srcX];
}
}