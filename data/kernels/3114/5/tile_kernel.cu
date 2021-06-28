#include "includes.h"
__global__ void tile_kernel(const float* in,float* out, int num_planes, int num_rows, int num_cols) {

const int gid = threadIdx.x + blockIdx.x * blockDim.x;
const int elems_per_plane = num_rows * num_cols;

const int plane = gid / num_rows;
const int row   = gid % num_rows;

if (plane >= num_planes)
return;

for (int col=0;col<num_cols; ++col){
out[plane * elems_per_plane + row * num_cols + col]=in[plane*num_cols+col];
}
}