#include "includes.h"










/*
* Implementations
*/
__global__ void ca_backward_kernel_f(const float *dw, const float *t, const float *f, float *df, int num, int chn, int height, int width) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int sp = height * width;
int len = height + width - 1;
int plane = blockIdx.z;

if (x < width && y < height && plane < chn) {

for (int batch = 0; batch < num; ++batch) {

for (int i = 0; i < width; ++i) {
float _dw = dw[(batch * len + x) * sp + y*width + i];
float _t = t[(batch * chn + plane) * sp + y*width + i];
df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
}
for (int i = 0; i < height; ++i) {
if (i == y) continue;
int j = i>y ? y : y-1;

float _dw = dw[(batch * len + width + j) * sp + i*width + x];
float _t = t[(batch * chn + plane) * sp + i*width + x];
df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
}
}

}
}