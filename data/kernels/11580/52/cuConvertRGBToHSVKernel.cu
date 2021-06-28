#include "includes.h"
__global__ void cuConvertRGBToHSVKernel(const float4* src, float4* dst, size_t stride, int width, int height, bool normalize)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int c = y*stride + x;

if (x<width && y<height)
{
// Read
float4 in = src[c];
float R = in.x;
float G = in.y;
float B = in.z;

float Ma = fmaxf(R, fmaxf(G, B));
float mi = fminf(R, fminf(G, B));
float C = Ma-mi;

// Hue
float H = 0.0f;
if (C != 0.0f)
{
if (Ma == R)
H = fmod((G - B)/C, 6.0f);
if (Ma == G)
H = (B - R)/C + 2.0f;
if (Ma == B)
H = (R - G)/C + 4.0f;
}

H *= 60.0f;

// Value
float V = Ma;

// Saturation
float S = 0.0f;
if (C != 0.0f)
S = C/V;

if (H < 0.0f)
H += 360.0f;

// Normalize
if (normalize)
H /= 360.0f;

// Write Back
dst[c] = make_float4(H, S, V, in.w);
}
}