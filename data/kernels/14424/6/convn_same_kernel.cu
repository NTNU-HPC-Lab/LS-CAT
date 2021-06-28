#include "includes.h"
__global__ void convn_same_kernel(float *output, float *data, float *kernel, const int H, const int W, const int kH, const int kW) {

// Matrix index
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x >= H || y >= W)
return;

const int i0 = kW / 2, j0 = kH / 2;

float sum = 0;
for (int i = 0; i < kW; ++i) {
for(int j = 0; j < kH; ++j) {
int ii = y - i + i0;
int jj = x - j + j0;

if ( ii < 0 || ii >= W || jj < 0 || jj >= H )
continue;

sum += kernel[ i * kH + j ] * data[ ii * H + jj ];
}
}

output[y * H + x] = sum;
}