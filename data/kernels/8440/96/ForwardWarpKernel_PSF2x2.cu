#include "includes.h"
__device__ __forceinline__ float imag(const float2& val)
{
return val.y;
}
__global__ void ForwardWarpKernel_PSF2x2(const float *u, const float *v, const float *src, const int w, const int h, const int flow_stride, const int image_stride, const float time_scale, float *normalization_factor, float *dst)
{
int j = threadIdx.x + blockDim.x * blockIdx.x;
int i = threadIdx.y + blockDim.y * blockIdx.y;

if (i >= h || j >= w) return;

int flow_row_offset  = i * flow_stride;
int image_row_offset = i * image_stride;

//bottom left corner of a target pixel
float cx = u[flow_row_offset + j] * time_scale + (float)j + 1.0f;
float cy = v[flow_row_offset + j] * time_scale + (float)i + 1.0f;
// pixel containing bottom left corner
float px;
float py;
float dx = modff (cx, &px);
float dy = modff (cy, &py);
// target pixel integer coords
int tx;
int ty;
tx = (int) px;
ty = (int) py;
float value = src[image_row_offset + j];
float weight;
// fill pixel containing bottom right corner
if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
{
weight = dx * dy;
_atomicAdd (dst + ty * image_stride + tx, value * weight);
_atomicAdd (normalization_factor + ty * image_stride + tx, weight);
}

// fill pixel containing bottom left corner
tx -= 1;
if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
{
weight = (1.0f - dx) * dy;
_atomicAdd (dst + ty * image_stride + tx, value * weight);
_atomicAdd (normalization_factor + ty * image_stride + tx, weight);
}

// fill pixel containing upper left corner
ty -= 1;
if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
{
weight = (1.0f - dx) * (1.0f - dy);
_atomicAdd (dst + ty * image_stride + tx, value * weight);
_atomicAdd (normalization_factor + ty * image_stride + tx, weight);
}

// fill pixel containing upper right corner
tx += 1;
if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
{
weight = dx * (1.0f - dy);
_atomicAdd (dst + ty * image_stride + tx, value * weight);
_atomicAdd (normalization_factor + ty * image_stride + tx, weight);
}
}