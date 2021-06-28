#include "includes.h"










/*
* Implementations
*/
__global__ void ca_backward_kernel_t(const float *dw, const float *t, const float *f, float *dt, int num, int chn, int height, int width) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int sp = height * width;
int len = height + width - 1;
int plane = blockIdx.z;

if (x < width && y < height && plane < chn) {
for (int batch = 0; batch < num; ++batch) {

for (int i = 0; i < width; ++i) {
float _dw = dw[(batch * len + i) * sp + y*width + x];
float _f = f[(batch * chn + plane) * sp + y*width + i];
dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
}
for (int i = 0; i < height; ++i)  {
if (i == y) continue;
int j = i<y ? i : i-1;

float _dw = dw[(batch * len + width + j) * sp + y*width + x];
float _f = f[(batch * chn + plane) * sp + i*width + x];
dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
}
}

}
}