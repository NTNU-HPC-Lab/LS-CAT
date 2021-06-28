#include "includes.h"
__global__ void kCopyToTransDestFast(float* srcStart, float* destStart, unsigned int srcCopyWidth, unsigned int srcCopyHeight, unsigned int srcJumpSize, unsigned int destJumpSize) {
//    const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
//    const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;

//    if(idxX < srcCopyWidth && idxY < srcCopyHeight) {
const unsigned int srcReadIdx = (blockIdx.y * blockDim.y + threadIdx.y) * srcJumpSize + blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int destWriteIdx =  (blockIdx.x * blockDim.x + threadIdx.y) * destJumpSize + blockIdx.y * blockDim.y + threadIdx.x;
__shared__ float smem[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE + 1];

smem[threadIdx.x][threadIdx.y] = srcStart[srcReadIdx];
__syncthreads();

destStart[destWriteIdx] = smem[threadIdx.y][threadIdx.x];
//    }
}