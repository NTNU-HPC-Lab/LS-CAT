#include "includes.h"
__global__ void __linComb(float *X, float wx, float *Y, float wy, float *Z, int len) {
int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
for (int i = ip; i < len; i += blockDim.x * gridDim.x * gridDim.y) {
Z[i] = X[i]*wx + Y[i]*wy;
}
}