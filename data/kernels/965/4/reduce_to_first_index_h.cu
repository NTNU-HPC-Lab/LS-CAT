#include "includes.h"
__global__ void reduce_to_first_index_h(float *X, int height, int width) {
int t = blockIdx.x * blockDim.x + threadIdx.x;
float tmp = 0;
if (t < width) {
for (int i = 0; i < height; i++) {
tmp += X[i * width + t];
}
X[t] = tmp;
}
}