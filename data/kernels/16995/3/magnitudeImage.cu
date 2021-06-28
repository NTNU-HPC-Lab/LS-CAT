#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void magnitudeImage(float *d_gradientX, float *d_gradientY, float *d_gradientMag, int width, int height) {
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;

if (ix < width && iy < height) {
int idx = iy * width + ix;
d_gradientMag[idx] = sqrtf(powf(d_gradientX[idx], 2.0f) + powf(d_gradientY[idx], 2.0f));
}
}