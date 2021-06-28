#include "includes.h"
__global__ void kTranspose(float* a, float* dest, int width, int height) {
const int bx = blockIdx.x * blockDim.x;
const int by = blockIdx.y * blockDim.y;
const int tx = bx + threadIdx.x;
const int ty = by + threadIdx.y;
//    unsigned int idx = ty * width + tx;

__shared__ float smem[ADD_BLOCK_SIZE][ADD_BLOCK_SIZE + 1];

if (tx < width && ty < height) {
smem[threadIdx.y][threadIdx.x] = a[ty * width + tx];
}
__syncthreads();

if (by + threadIdx.x < height && threadIdx.y + bx < width) {
//        idx = height * (blockIdx.x * blockDim.x + threadIdx.y) + blockIdx.y * blockDim.y + threadIdx.x;
dest[(bx + threadIdx.y) * height + by + threadIdx.x] = smem[threadIdx.x][threadIdx.y];
}
}