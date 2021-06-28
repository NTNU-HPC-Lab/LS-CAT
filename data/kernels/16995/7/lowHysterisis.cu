#include "includes.h"

# define MAX(a, b) ((a) > (b) ? (a) : (b))

# define GAUSSIAN_KERNEL_SIZE 3
# define SOBEL_KERNEL_SIZE 5
# define TILE_WIDTH 32
# define SMEM_SIZE 128
__global__ void lowHysterisis(int width, int height, float *d_nonMax, float* d_highThreshHyst, float lowThreshold, float *d_lowThreshHyst) {
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;

if ((ix > 0) && (ix < (width - 1)) && (iy > 0) && iy < (height - 1)) {
int tid = iy * width + ix;

d_lowThreshHyst[tid] = d_highThreshHyst[tid];

if (d_highThreshHyst[tid] == 1) {
// Determine neighbour indices
int eastN = tid + 1;
int westN = tid - 1;
int northN = tid - width;
int southN = tid + width;

int southEastN = southN + 1;
int northEastN = northN	+ 1;
int southWestN = southN - 1;
int northWestN = northN	- 1;

if (d_nonMax[eastN] > lowThreshold)
d_lowThreshHyst[eastN] = 1.0f;

if (d_nonMax[westN] > lowThreshold)
d_lowThreshHyst[westN] = 1.0f;

if (d_nonMax[northN] > lowThreshold)
d_lowThreshHyst[northN] = 1.0f;

if (d_nonMax[southN] > lowThreshold)
d_lowThreshHyst[southN] = 1.0f;

if (d_nonMax[southEastN] > lowThreshold)
d_lowThreshHyst[southEastN] = 1.0f;

if (d_nonMax[northEastN] > lowThreshold)
d_lowThreshHyst[northEastN] = 1.0f;

if (d_nonMax[southWestN] > lowThreshold)
d_lowThreshHyst[southWestN] = 1.0f;

if (d_nonMax[northWestN] > lowThreshold)
d_lowThreshHyst[northWestN] = 1.0f;
}
}
}