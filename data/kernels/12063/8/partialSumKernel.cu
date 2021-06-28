#include "includes.h"
__global__ void partialSumKernel(int *X, int N)
{
__shared__ int partialSum[2 * BLOCK_SIZE];
int tx = threadIdx.x;
int i = blockIdx.x * blockDim.x + tx;
partialSum[tx] = (i < N) ?  X[i] : 0;
partialSum[tx + blockDim.x] = 0;

for (int stride = blockDim.x; stride > 0; stride = stride/2)
{
__syncthreads();
if (tx <= stride) {
partialSum[tx] += partialSum[tx + stride];
//printf("tx[%d], bx[%d]: %d + %d\n", tx, blockIdx.x, partialSum[tx], partialSum[tx + stride]);
}
}
if (tx == 0)
X[blockIdx.x] = partialSum[tx];
}