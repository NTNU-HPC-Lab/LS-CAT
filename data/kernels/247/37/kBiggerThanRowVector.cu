#include "includes.h"
__global__ void kBiggerThanRowVector(float* mat, float* vec, float* tgtMat, const int width, const int height) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < width * height; i += numThreads) {
tgtMat[i] = mat[i] > vec[i % width];
}
}