#include "includes.h"
__device__ size_t GIDX(size_t row, size_t col, int H, int W) {
return row * W + col;
}
__global__ void kernel_sub(float* d_f1ptr, float* d_f2ptr, float* d_dt, int H, int W) {

size_t row = threadIdx.y + blockDim.y * blockIdx.y;
size_t col = threadIdx.x + blockDim.x * blockIdx.x;
size_t idx = GIDX(row, col, H, W);

if (row >= H || col >= W) {
return;
}

d_dt[idx] = d_f2ptr[idx] - d_f1ptr[idx];

}