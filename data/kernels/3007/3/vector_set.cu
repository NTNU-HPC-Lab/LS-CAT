#include "includes.h"
extern "C" {

#ifndef NUMBER
#define NUMBER float
#endif





}
__global__ void vector_set (const int n, const NUMBER val, NUMBER* x, const int offset_x, const int stride_x) {
const int gid = blockIdx.x * blockDim.x + threadIdx.x;
if (gid < n) {
x[offset_x + gid * stride_x] = val;
}
}