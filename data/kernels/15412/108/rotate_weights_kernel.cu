#include "includes.h"
__global__  void rotate_weights_kernel(const float *src_weight_gpu, float *weight_deform_gpu, int nweights, int n, int kernel_size, int reverse)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
const int kernel_area = kernel_size * kernel_size;
const int i = index * kernel_area;

const int stage_step = (nweights / kernel_area) / 4;  // 4 stages
const int stage_id = index / stage_step;

// nweights = (c / groups) * n * size * size;
// kernel_area = size*size

if (i < nweights)
{
// if(reverse)

if (stage_id == 0) {
// simple copy
for (int y = 0; y < kernel_size; ++y) {
for (int x = 0; x < kernel_size; ++x) {
const int src_i = x + y*kernel_size + i;
const int dst_i = x + y*kernel_size + i;
if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
}
}
}
else if (stage_id == 1)
{
// 90 degree clockwise rotation - 1
for (int y = 0; y < kernel_size; ++y) {
for (int x = 0; x < kernel_size; ++x) {
const int src_i = x + y*kernel_size + i;
const int dst_i = (kernel_size - 1 - y) + x*kernel_size + i;
if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
}
}
}
else if (stage_id == 2)
{
// 180 degree clockwise rotation - 2
for (int y = 0; y < kernel_size; ++y) {
for (int x = 0; x < kernel_size; ++x) {
const int src_i = x + y*kernel_size + i;
const int dst_i = (kernel_size - 1 - x) + (kernel_size - 1 - y)*kernel_size + i;
if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
}
}
}
else if (stage_id == 3)
{
// 270 degree clockwise rotation - 3
for (int y = 0; y < kernel_size; ++y) {
for (int x = 0; x < kernel_size; ++x) {
const int src_i = x + y*kernel_size + i;
const int dst_i = y + (kernel_size - 1 - x)*kernel_size + i;
if (reverse) weight_deform_gpu[src_i] = src_weight_gpu[dst_i];
else weight_deform_gpu[dst_i] = src_weight_gpu[src_i];
}
}
}
}
}