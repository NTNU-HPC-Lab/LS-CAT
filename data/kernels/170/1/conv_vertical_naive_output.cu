#include "includes.h"
__global__ void conv_vertical_naive_output(const int n, float *y, const float *x, const float *w, const int iH, const int iW, const int kL)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
int oH = iH - kL + 1;
int x_offset = (i/(oH*iW))*iH*iW + i%(oH*iW);
int w_offset = (i/(oH*iW))*kL;

for (int k = 0; k < kL; k++) {
y[i] += w[w_offset + k]*x[x_offset + k*iW];
}
}
}