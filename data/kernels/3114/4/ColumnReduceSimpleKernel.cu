#include "includes.h"
__global__ void ColumnReduceSimpleKernel(const float* in,float* out, int num_planes, int num_rows, int num_cols) {

const int gid = threadIdx.x + blockIdx.x * blockDim.x;
const int elems_per_plane = num_rows * num_cols;

const int plane = gid / num_cols;
const int col = gid % num_cols;

if (plane >= num_planes)
return;

float sum = in[plane * elems_per_plane + col]+in[plane * elems_per_plane + num_cols + col];
for (int row = 2; row < num_rows; ++row) {
sum = sum+in[plane * elems_per_plane + row * num_cols + col];
}
out[plane * num_cols + col] = sum;
}