#include "includes.h"
__global__ void conv_vertical_naive_gradInput(const int n, float *dx, const float *dy, const float *w, const int oH, const int oW, const int kL)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
int iH = oH + kL - 1;
int iC = i/(iH*oW);
int row = (i%(iH*oW))/oW;
int dy_offset = iC*oH*oW + i%(iH*oW);
int w_offset = iC*kL;

int k_begin = max(0, row-oH+1);
int k_end = min(kL, row+1);

dx[i] = 0.0f;
for (int k = k_begin; k < k_end; k++) {
dx[i] += w[w_offset + k]*dy[dy_offset - k*oW];
}
}
}