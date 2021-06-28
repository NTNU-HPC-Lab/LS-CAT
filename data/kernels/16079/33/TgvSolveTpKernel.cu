#include "includes.h"
__global__ void TgvSolveTpKernel(float*a, float *b, float*c, float2* p, float2* Tp, int width, int height, int stride) {
int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row
int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column

if ((iy < height) && (ix < width))
{
int pos = ix + iy * stride;

Tp[pos].x = a[pos] * p[pos].x + c[pos] * p[pos].y;
Tp[pos].y = c[pos] * p[pos].x + b[pos] * p[pos].y;
}
}