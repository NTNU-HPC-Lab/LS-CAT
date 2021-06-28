#include "includes.h"
extern "C" {

#ifndef NUMBER
#define NUMBER float
#endif





}
__global__ void vector_copy (const int n, const NUMBER* x, const int offset_x, const int stride_x, NUMBER* y, const int offset_y, const int stride_y) {

const int gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid < n) {
const int ix = offset_x + gid * stride_x;
const int iy = offset_y + gid * stride_y;
y[iy] = x[ix];
}
}