#include "includes.h"
__global__ void conv_vertical_naive_gradParam(const int n, float *dw, const float *x, const float *dy, const int kL, const int oH, const int oW)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
int dy_offset = (i/kL)*oH*oW;
int x_offset = (i/kL)*oH*oW + (i%kL)*oW;

for (int k = 0; k < oH*oW; k++) {
dw[i] += dy[dy_offset + k]*x[x_offset + k];
}
}
}