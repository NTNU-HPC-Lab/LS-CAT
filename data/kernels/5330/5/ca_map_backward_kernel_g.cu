#include "includes.h"










/*
* Implementations
*/
__global__ void ca_map_backward_kernel_g(const float *dout, const float *weight, const float *g, float *dg, int num, int chn, int height, int width) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int sp = height * width;
int len = height + width - 1;
int plane = blockIdx.z;

if (x < width && y < height && plane < chn) {

for (int batch = 0; batch < num; ++batch) {
for (int i = 0; i < width; ++i) {
float _dout = dout[(batch * chn + plane) * sp + y*width + i];
float _w = weight[(batch * len + x) * sp + y*width + i];
dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
}

for (int i = 0; i < height; ++i) {
if (i == y) continue;
int j = i>y ? y : y-1;

float _dout = dout[(batch * chn + plane) * sp + i*width + x];
float _w = weight[(batch * len + width + j) * sp + i*width + x];
dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
}
}
}
}