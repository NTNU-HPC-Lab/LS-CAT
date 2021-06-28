#include "includes.h"
__global__ void kAddColMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width * height; i += numThreads) {
tgtMat[i] = mat[i] + mult * vec[i % height];
}
}