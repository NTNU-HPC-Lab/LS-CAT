#include "includes.h"
__global__ void computeVertices_kernel(float4* pos, unsigned int width, unsigned int height, float time)
{
unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

// calculate uv coordinates
float u = x / (float) width;
float v = y / (float) height;
u = u*2.0 - 1.0f;
v = v*2.0 - 1.0f;

// calculate simple sine wave pattern
float freq = 4.0f;
float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

// write output vertex
pos[y*width+x] = make_float4(u, w, v, 1.0f);
}