#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void initializeSobel(float *d_sobelKernelX, float *d_sobelKernelY) {
int ix = threadIdx.x;
int iy = threadIdx.y;
int weight = SOBEL_KERNEL_SIZE / 2;

if (ix < SOBEL_KERNEL_SIZE && iy < SOBEL_KERNEL_SIZE) {
int index = iy * SOBEL_KERNEL_SIZE + ix;
float sx = ix - SOBEL_KERNEL_SIZE / 2;
float sy = iy - SOBEL_KERNEL_SIZE / 2;
float norm = sx * sx + sy *sy;

if (norm == 0.0f) {
d_sobelKernelX[index] = 0.0f;
d_sobelKernelY[index] = 0.0f;
}
else {
d_sobelKernelX[index] = sx * weight / norm;
d_sobelKernelY[index] = sy * weight / norm;
}
}
}