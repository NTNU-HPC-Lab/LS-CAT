#include "includes.h"
__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, unsigned int height) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
mat += idx;
if (idx < width) {
float sum = 0;
for (int j = 0; j < height; j++) {
sum += *mat;
mat += width;
}
vec[idx] = sum;
}
}