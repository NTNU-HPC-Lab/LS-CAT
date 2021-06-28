#include "includes.h"
__global__ void kSumRowwise(float* mat, float* target, unsigned int width, unsigned int height, float mult, float p) {
extern __shared__ float sum_vals[];
const int row = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
if (row < height) {
float sum = 0;
float *data = mat + row;
for (unsigned int i = 0; i < width; i++) sum += data[i*height];
__syncthreads();
target[row] = p * target[row] + mult * sum;
}
}