#include "includes.h"

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define GET_BLOCKS(n, t) (n+t-1) / t

// == Dimension rearrangement Kernel

__global__ void CorrelateDataSubtract_1d(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount, int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2, int bottomwidth, int bottomheight, int bottomchannels, const float *bottom0, const float *bottom1, float *top)
{
CUDA_KERNEL_LOOP(index, nthreads) {
int x = index % topwidth; //w-pos
int y = (index / topwidth) % topheight; //h-pos
int c = (index / topwidth / topheight) % topchannels; //channels

// Offset of patch in image 2
int s2o = (c % neighborhood_grid_width + x_shift) * stride2;

// First (upper left) position of kernel center in current neighborhood in image 1
int x1 = x*stride1 + kernel_radius + max_displacement;
int y1 = y*stride1 + kernel_radius + 0;

// Iterate through 3D patch
float sum = 0;
for(int j = -kernel_radius; j <= kernel_radius; j++) { // HEIGHT
for(int i = -kernel_radius; i <= kernel_radius; i++) { // WIDTH
for(int l = 0; l < bottomchannels; l++) { // CHANNELS
// Calculate position in image 2
int x2 = x1 + s2o;
int y2 = y1;

// Indices in bottom data: (CH=l,W=x2,H=y2,N)
int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + l;
int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + l;

// Do the correlation:
sum += fabsf(bottom0[idx1] - bottom1[idx2]);
}
}
}
const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
top[index + item*topcount] = sum / (float)sumelems;
}

}