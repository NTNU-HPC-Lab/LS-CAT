#include "includes.h"
__global__ void kTile(const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight) {
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
const int numThreads = blockDim.x * gridDim.x;
//    const unsigned int numEls = tgtWidth * tgtHeight;
for (uint i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
const uint y = i / tgtWidth;
const uint x = i % tgtWidth;
const uint srcY = y % srcHeight;
const uint srcX = x % srcWidth;
tgt[i] = src[srcY * srcWidth + srcX];
}
}