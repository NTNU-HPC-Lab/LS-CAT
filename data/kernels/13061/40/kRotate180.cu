#include "includes.h"
__global__ void kRotate180(float* filters, float* targets, const int filterSize) {
//   __shared__ float shFilter[16][16];

const int filtIdx = blockIdx.x;
const int readStart = MUL24(MUL24(filterSize, filterSize), filtIdx);
filters += readStart;
targets += readStart;

for(int y = threadIdx.y; y < filterSize; y += 16) {
for(int x = threadIdx.x; x < filterSize; x += 16) {
const int writeX = filterSize - 1 - x;
const int writeY = filterSize - 1 - y;

targets[MUL24(writeY, filterSize) + writeX] = filters[MUL24(y, filterSize) + x];
}
}
}