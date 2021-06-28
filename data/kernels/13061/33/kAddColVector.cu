#include "includes.h"
__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, const unsigned int width, const unsigned int height, const float scaleVec) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < width * height; i += numThreads) {
tgtMat[i] = mat[i] + scaleVec * vec[i / width];
}
}