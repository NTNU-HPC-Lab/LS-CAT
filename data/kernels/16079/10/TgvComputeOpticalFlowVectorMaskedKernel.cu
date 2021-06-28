#include "includes.h"
__global__ void TgvComputeOpticalFlowVectorMaskedKernel(const float *u, const float2 *tv2, float* mask, int width, int height, int stride, float2 *warpUV)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

if ((iy >= height) && (ix >= width)) return;
int pos = ix + iy * stride;
if (mask[pos] == 0.0f) return;

float us = u[pos];
float2 tv2s = tv2[pos];
warpUV[pos].x = us * tv2s.x;
warpUV[pos].y = us * tv2s.y;
}