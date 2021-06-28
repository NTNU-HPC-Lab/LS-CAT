#include "includes.h"










/*
* Implementations
*/
__global__ void ca_forward_kernel(const float *t, const float *f, float *weight, int num, int chn, int height, int width) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int sp = height * width;
int len = height + width - 1;
int z = blockIdx.z;

if (x < width && y < height && z < height+width-1) {
for (int batch = 0; batch < num; ++batch) {
for (int plane = 0; plane < chn; ++plane) {
float _t = t[(batch * chn + plane) * sp + y*width + x];

if (z < width) {
int i = z;
float _f = f[(batch * chn + plane) * sp + y*width + i];
weight[(batch * len + i) * sp + y*width + x] += _t*_f;
} else {
int i = z - width;
int j = i<y ? i : i+1;

float _f = f[(batch * chn + plane) * sp + j*width + x];
weight[(batch * len + width + i) * sp + y*width + x] += _t*_f;
}
}
}
}
}