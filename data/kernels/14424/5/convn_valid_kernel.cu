#include "includes.h"
__global__ void convn_valid_kernel(float *output, float *data, float *kernel, const int H, const int W, const int kH, const int kW) {

// Matrix index
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

// vH, vW stands for valid H and valid W
const int vH = H - kH + 1;
const int vW = W - kW + 1;

if (x >= vH || y >= vW)
return;

x += kH - 1;
y += kW - 1;

float sum = 0;
for (int i = 0; i < kW; ++i)
for(int j = 0; j < kH; ++j)
sum += kernel[ i * kH + j ] * data[ (y - i) * H + (x - j) ];

x -= kH - 1;
y -= kW - 1;

output[ y * vH + x ] = sum;
}