#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 1e-4

__global__ void bilateral_smooth_kernel( float *affine_model, float *filtered_affine_model, float *guide, int h, int w, int kernel_radius, float sigma1, float sigma2 )
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
int size = h * w;
if (id < size) {
int x = id % w;
int y = id / w;

double sum_affine[12] = {};
double sum_weight = 0;
for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
int yy = y + dy, xx = x + dx;
int id2 = yy * w + xx;
if (0 <= xx && xx < w && 0 <= yy && yy < h) {
float color_diff1 = guide[yy*w + xx] - guide[y*w + x];
float color_diff2 = guide[yy*w + xx + size] - guide[y*w + x + size];
float color_diff3 = guide[yy*w + xx + 2*size] - guide[y*w + x + 2*size];
float color_diff_sqr =
(color_diff1*color_diff1 + color_diff2*color_diff2 + color_diff3*color_diff3) / 3;

float v1 = exp(-(dx * dx + dy * dy) / (2 * sigma1 * sigma1));
float v2 = exp(-(color_diff_sqr) / (2 * sigma2 * sigma2));
float weight = v1 * v2;

for (int i = 0; i < 3; i++) {
for (int j = 0; j < 4; j++) {
int affine_id = i * 4 + j;
sum_affine[affine_id] += weight * affine_model[id2*12 + affine_id];
}
}
sum_weight += weight;
}
}
}

for (int i = 0; i < 3; i++) {
for (int j = 0; j < 4; j++) {
int affine_id = i * 4 + j;
filtered_affine_model[id*12 + affine_id] = sum_affine[affine_id] / sum_weight;
}
}
}
return ;
}