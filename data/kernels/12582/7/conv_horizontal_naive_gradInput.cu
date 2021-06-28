#include "includes.h"
__global__ void conv_horizontal_naive_gradInput(const int n, float *dx, const float *dy, const float *w, const int oH, const int oW, const int kL)
{
for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
int iW = oW + kL - 1;
int col = i%iW;
int dy_offset = (i/iW)*oW + i%iW;
int w_offset = (i/(iW*oH))*kL;

int k_begin = max(0, col-oW+1);
int k_end = min(kL, col+1);

dx[i] = 0.0f;
for (int k = k_begin; k < k_end; k++) {
dx[i] += w[w_offset + k]*dy[dy_offset - k];
}
}
}