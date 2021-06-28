#include "includes.h"
__global__ void kMultByRowVectorScale(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height, float scale_targets) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width * height; i += numThreads) {
tgtMat[i] = scale_targets * tgtMat[i] + mat[i] * vec[i / height];
}
}