#include "includes.h"
__global__ void kDotProduct_r(float* a, float* b, float* target, const uint numCols, const uint numElements) {
__shared__ float shmem[DP_BLOCKSIZE];

uint eidx = DP_BLOCKSIZE * blockIdx.x + threadIdx.x;
shmem[threadIdx.x] = 0;
if (eidx < numCols) {
for (; eidx < numElements; eidx += numCols) {
shmem[threadIdx.x] += a[eidx] * b[eidx];
}
}
__syncthreads();
if (threadIdx.x < 256) {
shmem[threadIdx.x] += shmem[threadIdx.x + 256];
}
__syncthreads();
if (threadIdx.x < 128) {
shmem[threadIdx.x] += shmem[threadIdx.x + 128];
}
__syncthreads();
if (threadIdx.x < 64) {
shmem[threadIdx.x] += shmem[threadIdx.x + 64];
}
__syncthreads();
if (threadIdx.x < 32) {
volatile float* mysh = &shmem[threadIdx.x];
*mysh += mysh[32];
*mysh += mysh[16];
*mysh += mysh[8];
*mysh += mysh[4];
*mysh += mysh[2];
*mysh += mysh[1];
if (threadIdx.x == 0) {
target[blockIdx.x] = *mysh;
}
}
}