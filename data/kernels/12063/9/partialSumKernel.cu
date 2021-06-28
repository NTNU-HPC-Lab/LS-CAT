#include "includes.h"
__global__ void partialSumKernel(int *X, int N)
{
__shared__ int partialSum[BLOCK_SIZE];
int tx = threadIdx.x;
int i = blockIdx.x * blockDim.x + tx;

if (i < N) {
partialSum[tx] = X[i];
partialSum[tx + blockDim.x] = X[i + gridDim.x * blockDim.x];
//printf("X[%d + %d * %d] = %d\n", i,gridDim.x, blockDim.x, X[i + gridDim.x * blockDim.x]);
}
else
partialSum[tx] = 0; // last block may pad with 0's

for (int stride = blockDim.x; stride > 0; stride = stride/2)
{
__syncthreads();
if (tx < stride) {
//printf("tx[%d], bx[%d]: %d + %d\n", tx, blockIdx.x, partialSum[tx], partialSum[tx + stride]);
partialSum[tx] += partialSum[tx + stride];
}
}
if (tx == 0)
X[blockIdx.x] = partialSum[tx];
}