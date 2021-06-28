#include "includes.h"
__global__ void cuConvertLABToRGBKernel(const float4* src, float4* dst, size_t stride, int width, int height)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int c = y*stride + x;

if (x<width && y<height)
{
// Read
float4 in = src[c];

float L = in.x;
float a = in.y;
float b = in.z;


// convert to XYZ
const float T1 = cbrtf(216/24389.0f);
const float fy = (L+16) / 116.0f;

float4 XYZ;
if (L > 8)
XYZ.y = fy*fy*fy;
else
XYZ.y = L / (24389/27.0f);

float fx = a/500.0f + fy;
if (fx > T1)
XYZ.x = fx*fx*fx;
else
XYZ.x = (116*fx-16) / (24389/27.0f);

float fz = fy - b/200.0f;
if (fz > T1)
XYZ.z = fz*fz*fz;
else
XYZ.z = (116*fz-16) / (24389/27.0f);


// Normalize for D65 white point
XYZ.x *= 0.950456f;
XYZ.z *= 1.088754f;

float4 rgb;
rgb.x = 3.2404542f*XYZ.x + -1.5371385f*XYZ.y + -0.4985314f*XYZ.z;
rgb.y = -0.9692660f*XYZ.x + 1.8760108f*XYZ.y + 0.0415560f*XYZ.z;
rgb.z = 0.0556434f*XYZ.x + -0.2040259f*XYZ.y + 1.0572252f*XYZ.z;
rgb.w = in.w;

dst[c] = rgb;
}
}