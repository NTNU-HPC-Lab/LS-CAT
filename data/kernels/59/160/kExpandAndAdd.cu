#include "includes.h"
__global__ void kExpandAndAdd(float* source, float* mat, float* indices, float* target, int width, int height, float mult, int width2){
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width*height; i += numThreads) {
const int pos = height * (int)indices[i / height] + i % height;
target[i] = (pos < height * width2)? source[i] + mult * mat[pos] : 1.0/0.0 - 1.0/0.0;
}
}