#include "includes.h"
__global__ void conv_horizontal_naive_output(const int n, float *y, const float *x, const float *w, const int iH, const int iW, const int kL)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
int oW = iW - kL + 1;
int x_offset = (i/oW)*iW + i%oW;
int w_offset = (i/(oW*iH))*kL;

for (int k = 0; k < kL; k++) {
y[i] += w[w_offset + k]*x[x_offset + k];
}
}
}