#include "includes.h"
__global__ void kDivideTransFast(float* a, float* b, float* dest, unsigned int width, unsigned int height, unsigned int bJumpWidth) {
const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int idx = idxY * width + idxX;

__shared__ float smem[ADD_BLOCK_SIZE][ADD_BLOCK_SIZE + 1];

const unsigned int bBlockReadStart = blockDim.x * blockIdx.x * bJumpWidth + blockIdx.y * blockDim.y;

smem[threadIdx.x][threadIdx.y] = b[bBlockReadStart + threadIdx.y * bJumpWidth + threadIdx.x];
__syncthreads();

dest[idx] = __fdividef(a[idx], smem[threadIdx.y][threadIdx.x]);
}