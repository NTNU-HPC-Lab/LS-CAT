#include "includes.h"
__global__ void ComputeDerivativesKernel(int width, int height, int stride, float* Ix, float* Iy, float* Iz, cudaTextureObject_t texSource, cudaTextureObject_t texTarget)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;


if (ix >= width || iy >= height) return;

float dx = 1.0f / (float)width;
float dy = 1.0f / (float)height;

float x = ((float)ix + 0.5f) * dx;
float y = ((float)iy + 0.5f) * dy;

float t0, t1;
// x derivative
t0 = tex2D<float>(texSource, x + 2.0f * dx, y);
t0 -= tex2D<float>(texSource, x + 1.0f * dx, y) * 8.0f;
t0 += tex2D<float>(texSource, x - 1.0f * dx, y) * 8.0f;
t0 -= tex2D<float>(texSource, x - 2.0f * dx, y);
t0 /= 12.0f;

t1 = tex2D<float>(texTarget, x + 2.0f * dx, y);
t1 -= tex2D<float>(texTarget, x + 1.0f * dx, y) * 8.0f;
t1 += tex2D<float>(texTarget, x - 1.0f * dx, y) * 8.0f;
t1 -= tex2D<float>(texTarget, x - 2.0f * dx, y);
t1 /= 12.0f;

*(((float*)((char*)Ix + stride * iy)) + ix) = (t0 + t1) * 0.5f;

// t derivative
*(((float*)((char*)Iz + stride * iy)) + ix) = tex2D<float>(texSource, x, y) - tex2D<float>(texTarget, x, y);

// y derivative
t0 = tex2D<float>(texSource, x, y + 2.0f * dy);
t0 -= tex2D<float>(texSource, x, y + 1.0f * dy) * 8.0f;
t0 += tex2D<float>(texSource, x, y - 1.0f * dy) * 8.0f;
t0 -= tex2D<float>(texSource, x, y - 2.0f * dy);
t0 /= 12.0f;

t1 = tex2D<float>(texTarget, x, y + 2.0f * dy);
t1 -= tex2D<float>(texTarget, x, y + 1.0f * dy) * 8.0f;
t1 += tex2D<float>(texTarget, x, y - 1.0f * dy) * 8.0f;
t1 -= tex2D<float>(texTarget, x, y - 2.0f * dy);
t1 /= 12.0f;

*(((float*)((char*)Iy + stride * iy)) + ix) = (t0 + t1) * 0.5f;
}