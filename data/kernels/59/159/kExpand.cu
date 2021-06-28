#include "includes.h"
__global__ void kExpand(float* source, float* indices, float* target, int height, int width, int target_width){
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < target_width*height; i += numThreads) {
const int pos = height * (int)indices[i / height] + i % height;
target[i] = (pos < height * width)? source[pos] : 1.0/0.0 - 1.0/0.0;
}
}