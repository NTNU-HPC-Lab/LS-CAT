#include "includes.h"
__global__ void TgvComputeOpticalFlowVectorKernel(const float *u, const float2 *tv2, int width, int height, int stride, float2 *warpUV)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

const int pos = ix + iy * stride;

if (ix >= width || iy >= height) return;

float us = u[pos];
float2 tv2s = tv2[pos];
warpUV[pos].x = us * tv2s.x;
warpUV[pos].y = us * tv2s.y;
}