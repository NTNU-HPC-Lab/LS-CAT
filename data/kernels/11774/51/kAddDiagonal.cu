#include "includes.h"
__global__ void kAddDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < width; i += numThreads) {
tgtMat[width*i + i] = mat[width*i + i] + vec[i];
}
}