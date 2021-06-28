#include "includes.h"
__device__ size_t GIDX(size_t row, size_t col, int H, int W) {
return row * W + col;
}
__global__ void kernel_blur(float* d_I, float* d_Ib, int H, int W) {

size_t row = threadIdx.y + blockDim.y * blockIdx.y;
size_t col = threadIdx.x + blockDim.x * blockIdx.x;
size_t idx = GIDX(row, col, H, W);

if (row >= H - KERN_RADIUS || row <= KERN_RADIUS || col >= W - KERN_RADIUS || col <= KERN_RADIUS) {
return;
}

int count = 0;
for (int i = -KERN_RADIUS; i <= KERN_RADIUS; i++) {
for (int j = -KERN_RADIUS; j <= KERN_RADIUS; j++) {
d_Ib[idx] += d_I[GIDX(row + i, col + j, H, W)] * gaussian_kernel[count];
count++;
}
}

}