#include "includes.h"

__global__ void UpdateSurface(cudaSurfaceObject_t surf, unsigned int width, unsigned int height, float time)
{
unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
if (y >= height | x >= width) return;

auto xVar = (float)x / (float)width;
auto yVar = (float)y / (float)height;
auto cost = __cosf(time) * 0.5f + 0.5f;
auto costx = __cosf(time) * 0.5f + xVar;
auto costy = __cosf(time) * 0.5f + yVar;
auto costxx = (__cosf(time) * 0.5f + 0.5f) * width;
auto costyy = (__cosf(time) * 0.5f + 0.5f) * height;
auto costxMany = __cosf(y * time) * 0.5f + yVar;
auto costyMany = __cosf((float)x/100 * time) * 0.5f + xVar;
auto margin = 1;

float4 pixel{};
if (y == 0)
pixel = make_float4(costyMany * 0.3, costyMany * 1, costyMany * 0.4, 1);
else if (y == height - 1)
pixel = make_float4(costyMany * 0.6, costyMany * 0.7, costyMany * 1, 1);
else if (x % 2 == 0)
{
if (x > width / 2)
pixel = make_float4(0.1, 0.5, costx * 1, 1);
else
pixel = make_float4(costx * 1, 0.1, 0.2, 1);
}
else if (x > width - margin - 1 | x <= margin)
pixel = make_float4(costxMany, costxMany * 0.9, costxMany * 0.6, 1);
else
pixel = make_float4(costx * 0.3, costx * 0.4, costx * 0.6, 1);
surf2Dwrite(pixel, surf, x * 16, y);
}