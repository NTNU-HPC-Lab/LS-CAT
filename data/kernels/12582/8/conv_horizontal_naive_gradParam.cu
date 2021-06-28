#include "includes.h"
__global__ void conv_horizontal_naive_gradParam(const int n, float *dw, const float *x, const float *dy, const int kL, const int oH, const int oW)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
int iW = oW + kL - 1;
int dy_offset = (i/kL)*oH*oW;
int x_offset = (i/kL)*oH*oW + i%kL;

for (int j = 0; j < oH; j++) {
for (int k = 0; k < oW; k++) {
dw[i] += dy[dy_offset + j*oW + k]*x[x_offset + j*iW + k];
}
}
}
}