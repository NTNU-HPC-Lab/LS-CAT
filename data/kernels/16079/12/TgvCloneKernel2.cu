#include "includes.h"
__global__ void TgvCloneKernel2(float2* dst, float2* src, int width, int height, int stride) {
int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row
int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column

if ((iy < height) && (ix < width))
{
int pos = ix + iy * stride;
dst[pos] = src[pos];
}
}