#include "includes.h"
//============================================================================
// Name        : CudaMap.cu
// Author      : Hang
//============================================================================



using namespace std;


__global__ void addTen(float* d, int count) {

int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;

// Thread position in the block
int threadPosInBlock = threadIdx.x + blockDim.x * threadIdx.y +
blockDim.x * blockDim.y * threadIdx.z;

// Block position in grid
int blockPosInGrid = blockIdx.x + gridDim.x * blockIdx.y +
gridDim.x * gridDim.y * blockIdx.z;

// Final thread ID
int tid = blockPosInGrid * threadsPerBlock + threadPosInBlock;

if (tid < count) {
d[tid] = d[tid] + 10;
}

}