#include "includes.h"



__global__ void GaussianKernelSimple(const uint8_t *src, uint8_t *dst, int width, int height, int step)
{
const float filter[5][5] = {
{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
{ 0.021938231f, 0.098320331f, 0.162102822f, 0.098320331f, 0.021938231f },
{ 0.01330621f, 0.059634295f, 0.098320331f, 0.059634295f, 0.01330621f },
{ 0.002969017f, 0.01330621f, 0.021938231f, 0.01330621f, 0.002969017f },
};

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < width && y < height) {
float sum = 0;
for (int dy = 0; dy < 5; ++dy) {
for (int dx = 0; dx < 5; ++dx) {
sum += filter[dy][dx] * src[(x + dx) + (y + dy) * step];
}
}
dst[x + y * step] = (int)(sum + 0.5f);
}
}