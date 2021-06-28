#include "includes.h"
__global__ void cudaKernelMesh(float4* pos, unsigned int width, unsigned int height, float time)
{
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

// calculate uv coordinates
float u = x / (float) width;
float v = y / (float) height;
u = u*2.0f - 1.0f;
v = v*2.0f - 1.0f;

// calculate simple sine wave pattern
float freq = 4.0f;
float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

// write output vertex
pos[y*width+x] = make_float4(u, w, v, __int_as_float(0xff00ff00)); //Color : DirectX ARGB, OpenGL ABGR
}