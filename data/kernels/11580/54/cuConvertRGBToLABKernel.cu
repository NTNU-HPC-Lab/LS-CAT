#include "includes.h"
__global__ void cuConvertRGBToLABKernel(const float4* src, float4* dst, size_t stride, int width, int height, bool isNormalized)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int c = y*stride + x;

if (x<width && y<height)
{
// Read
float4 in = src[c];
if (!isNormalized)
{
in.x /= 255.0f;
in.y /= 255.0f;
in.z /= 255.0f;
in.w /= 255.0f;
}

float R = in.x;
float G = in.y;
float B = in.z;


// convert to XYZ
float4 XYZ;
XYZ.x = 0.4124564f*R + 0.3575761f*G + 0.1804375f*B;
XYZ.y = 0.2126729f*R + 0.7151522f*G + 0.0721750f*B;
XYZ.z = 0.0193339f*R + 0.1191920f*G + 0.9503041f*B;

// normalize for D65 white point
XYZ.x /= 0.950456f;
XYZ.z /= 1.088754f;

float cubeRootX, cubeRootY, cubeRootZ;
const float T1 = 216/24389.0f;
const float T2 = 24389/27.0f;

if (XYZ.x > T1)
cubeRootX = cbrtf(XYZ.x);
else
cubeRootX = (T2 * XYZ.x + 16) / 116;

if (XYZ.y > T1)
cubeRootY = cbrtf(XYZ.y);
else
cubeRootY = (T2 * XYZ.y + 16) / 116;

if (XYZ.z > T1)
cubeRootZ = cbrtf(XYZ.z);
else
cubeRootZ = (T2 * XYZ.z + 16) / 116;



dst[c] = make_float4(116*cubeRootY-16, 500*(cubeRootX-cubeRootY), 200*(cubeRootY-cubeRootZ), in.w);
}
}