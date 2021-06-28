#include "includes.h"
__global__ void gpu_colorRampHeatMapUnsat(uchar3 * colored, const float * vals, const int width, const int height, const float minVal, const float maxVal) {

const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x >= width || y >= height) {
return;
}

const int index = x + y*width;
uchar3 & imgVal = colored[index];

if (isnan(vals[index])) {
imgVal = make_uchar3(255,255,255);
return;
}

const float normVal = fmaxf(0,fminf((vals[index] - minVal)/(maxVal-minVal),1));

const float t = normVal == 1.0 ? 1.0 : fmodf(normVal,0.25)*4;
uchar3 a, b;
if (normVal < 0.25) { b = make_uchar3(32,191,139); a = make_uchar3(0x18,0x62,0x93); }
else if (normVal < 0.5) { b = make_uchar3(241,232,137); a = make_uchar3(32,191,139); }
else if (normVal < 0.75) { b = make_uchar3(198,132,63); a = make_uchar3(241,232,137); }
else { b = make_uchar3(0xc0,0x43,0x36); a = make_uchar3(198,132,63); }
imgVal = make_uchar3((1-t)*a.x + t*b.x,
(1-t)*a.y + t*b.y,
(1-t)*a.z + t*b.z);

}