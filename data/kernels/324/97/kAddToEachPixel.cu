#include "includes.h"
__global__ void kAddToEachPixel(float* mat1, float* mat2, float* tgtMat, float mult, unsigned int width, unsigned int height, unsigned int num_pix) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width * height; i += numThreads) {
tgtMat[i] = mat1[i] + mult * mat2[i % height + height * (i / (height * num_pix))];
}
}