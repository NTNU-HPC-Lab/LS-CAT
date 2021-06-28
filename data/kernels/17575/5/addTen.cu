#include "includes.h"
__global__ void addTen(float* d, int count) {
int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
int threadPosInBlock = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
int blockPosInGrid = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

int tid = blockPosInGrid * threadsPerBlock + threadPosInBlock;

if(tid < count) {
d[tid] = d[tid] + 10;
}
}