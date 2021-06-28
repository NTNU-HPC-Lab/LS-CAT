#include "includes.h"










/*
* Implementations
*/
__global__ void ca_map_backward_kernel_w(const float *dout, const float *weight, const float *g, float *dw, int num, int chn, int height, int width) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int sp = height * width;
int len = height + width - 1;
int z = blockIdx.z;

if (x < width && y < height && z < height+width-1) {

for (int batch = 0; batch < num; ++batch) {
for (int plane = 0; plane < chn; ++plane) {
float _dout = dout[(batch * chn + plane) * sp + y*width + x];

if (z < width) {
int i = z;
float _g = g[(batch * chn + plane) * sp + y*width + i];
dw[(batch * len + i) * sp + y*width + x] += _dout * _g;
} else {
int i = z - width;
int j = i<y ? i : i+1;

float _g = g[(batch * chn + plane) * sp + j*width + x];
dw[(batch * len + width + i) * sp + y*width + x] += _dout * _g;
}
}
}
}
}