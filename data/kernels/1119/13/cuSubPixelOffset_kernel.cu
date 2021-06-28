#include "includes.h"
__global__ void cuSubPixelOffset_kernel(const int2 *offsetInit, const int2 *offsetZoomIn, float2 *offsetFinal, const float OSratio, const float xoffset, const float yoffset, const int size)
{
int idx = threadIdx.x + blockDim.x*blockIdx.x;
if (idx >= size) return;
offsetFinal[idx].x = OSratio*(offsetZoomIn[idx].x ) + offsetInit[idx].x  - xoffset;
offsetFinal[idx].y = OSratio*(offsetZoomIn[idx].y ) + offsetInit[idx].y - yoffset;
}