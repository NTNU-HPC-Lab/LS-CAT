#include "includes.h"
__global__ void convertFlowToRGBA_kernel(uchar4 *d_flowx_out, uchar4 *d_flowy_out, const float *d_flowx_in, const float *d_flowy_in, int width, int height, float lowerLim, float upperLim, float minMag) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
uchar4 tempx, tempy;
if (x < width && y < height) {
float ux = d_flowx_in[__mul24(y, width) + x];
float uy = d_flowy_in[__mul24(y, width) + x];

float mag = sqrtf(ux * ux + uy * uy);

// first draw unmatched pixels in white
if (!isfinite(ux) || (mag < minMag)) {

tempx.x = 255;
tempx.y = 255;
tempx.z = 255;
tempx.w = 255;
tempy.x = 255;
tempy.y = 255;
tempy.z = 255;
tempy.w = 255;

} else {

// rescale value from [lowerLim,upperLim] to [0,1]
ux -= lowerLim;
ux /= (upperLim - lowerLim);

float r = 1.0f;
float g = 1.0f;
float b = 1.0f;
if (ux < 0.25f) {
r = 0;
g = 4.0f * ux;
} else if (ux < 0.5f) {
r = 0;
b = 1.0 + 4.0f * (0.25f - ux);
} else if (ux < 0.75f) {
r = 4.0f * (ux - 0.5f);
b = 0;
} else {
g = 1.0f + 4.0f * (0.75f - ux);
b = 0;
}
tempx.x = 255.0 * r;
tempx.y = 255.0 * g;
tempx.z = 255.0 * b;
tempx.w = 255;

uy -= lowerLim;
uy /= (upperLim - lowerLim);

r = 1.0f;
g = 1.0f;
b = 1.0f;
if (uy < 0.25f) {
r = 0;
g = 4.0f * uy;
} else if (uy < 0.5f) {
r = 0;
b = 1.0 + 4.0f * (0.25f - uy);
} else if (uy < 0.75f) {
r = 4.0f * (uy - 0.5f);
b = 0;
} else {
g = 1.0f + 4.0f * (0.75f - uy);
b = 0;
}
tempy.x = 255.0 * r;
tempy.y = 255.0 * g;
tempy.z = 255.0 * b;
tempy.w = 255;
}

d_flowx_out[__mul24(y, width) + x] = tempx;
d_flowy_out[__mul24(y, width) + x] = tempy;
}
}