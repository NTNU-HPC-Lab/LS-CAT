#include "includes.h"
__global__ void write_to_surface(const float *data, cudaSurfaceObject_t surface, const int width, const int height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
const int b = 4 * blockIdx.z;

if (x < width && y < height) {
const int wh = width * height;
const int offset = b * wh + y * width + x;

float4 tmp;
tmp.x = data[0 * wh + offset];
tmp.y = data[1 * wh + offset];
tmp.z = data[2 * wh + offset];
tmp.w = data[3 * wh + offset];

surf2DLayeredwrite<float4>(tmp, surface, x * sizeof(float4), y, blockIdx.z);
}
}