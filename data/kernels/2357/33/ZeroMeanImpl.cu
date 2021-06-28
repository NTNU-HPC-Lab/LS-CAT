#include "includes.h"
__global__ void ZeroMeanImpl(float* solutions, int rowSize, int matCount) {

const int matricesPerBlock = BLOCK_SIZE / rowSize;

const int matrixIdx = blockIdx.x * matricesPerBlock + threadIdx.x / rowSize;
const int tid = threadIdx.x;
const int col = threadIdx.x & (rowSize - 1);
const int inBlockOffset = threadIdx.x / rowSize;

__shared__ double beta[BLOCK_SIZE];
__shared__ double line[BLOCK_SIZE];

if (matrixIdx >= matCount) {
return;
}

solutions += matrixIdx * rowSize;
beta[tid] = col != (rowSize - 1) ? solutions[col] : 0;
line[tid] = beta[tid];
__syncthreads();

for (int s = rowSize >> 1; s > 0; s >>= 1) {
if (col < s) {
line[tid] += line[tid + s];
}
__syncthreads();
}

beta[tid] -= line[rowSize * inBlockOffset] / rowSize;
solutions[col] = beta[tid];
}