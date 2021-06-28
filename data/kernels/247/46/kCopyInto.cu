#include "includes.h"
__global__ void kCopyInto(float* images, float* targets, const int imgSize, const int paddingSize, const int numImages) {
const int imgIdx = blockIdx.y * gridDim.x + blockIdx.x;
if (imgIdx < numImages) {
const int targetSize = imgSize + 2 * paddingSize;
images += imgIdx * imgSize * imgSize;
targets += imgIdx * targetSize * targetSize + MUL24(paddingSize, targetSize) + paddingSize;
for (int y = threadIdx.y; y < imgSize; y += 16) {
for (int x = threadIdx.x; x < imgSize; x += 16) {
targets[MUL24(y, targetSize) + x] = images[MUL24(y, imgSize) + x];
}
}
}
}