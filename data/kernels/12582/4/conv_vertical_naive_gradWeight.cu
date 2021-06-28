#include "includes.h"
__global__ void conv_vertical_naive_gradWeight(const int n, float *y, const float *x, const int kL, const int iC)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
y[i] = x[(i/kL)*kL*iC + i];
}
}