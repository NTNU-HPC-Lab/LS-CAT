#include "includes.h"
__global__ void convn_full_kernel(float *output, float *data, float *kernel, int H, int W, int kH, int kW) {

// Matrix index
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

// fH, fW stands for full H and full W
const int fH = H + kH - 1;
const int fW = W + kW - 1;

if (x >= fH || y >= fW)
return;

float sum = 0;
for (int i = 0; i < kW; ++i) {
for(int j = 0; j < kH; ++j) {
int ii = y - i;
int jj = x - j;

if ( ii < 0 || ii >= W || jj < 0 || jj >= H )
continue;

sum += kernel[ i * kH + j ] * data[ ii * H + jj ];
}
}

output[ y * fH + x ] = sum;
}