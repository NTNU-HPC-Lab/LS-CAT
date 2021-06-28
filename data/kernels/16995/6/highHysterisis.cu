#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void highHysterisis(int width, int height, float* d_nonMax, float highThreshold, float *d_highThreshHyst) {
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;

if (ix < width && iy < height) {
int tid = iy * width + ix;

d_highThreshHyst[tid] = 0.0f;
if(d_nonMax[tid] > highThreshold)
d_highThreshHyst[tid] = 1.0f;
}
}