#include "includes.h"
extern "C" {

#ifndef DTYPE
#define DTYPE float
#endif






}
__global__ void tensor_5d_equals (const int n, const int c, const int d, const int h, const int w, const DTYPE* x, const int offset_x, const int n_x, const int c_x, const int d_x, const int h_x, const int w_x, const DTYPE* y, const int offset_y, const int n_y, const int c_y, const int d_y, const int h_y, const int w_y, int* eq_flag) {
const int gid_n = blockIdx.x * blockDim.x + threadIdx.x;
const int gid_c = blockIdx.y * blockDim.y + threadIdx.y;
const int gid_d = blockIdx.z * blockDim.z + threadIdx.z;
const bool valid = (gid_n < n) && (gid_c < c) && (gid_d < d);
if (valid) {
const int ix = offset_x + gid_n * n_x + gid_c * c_x + gid_d * d_x;
const int iy = offset_y + gid_n * n_y + gid_c * c_y + gid_d * d_y;
for (int i = 0; i < h; i++) {
for (int j = 0; j < w; j++) {
if (x[ix + i * h_x + j * w_x] != y[iy + i * h_y + j * w_y]){
eq_flag[0]++;
}
}
};
}
}