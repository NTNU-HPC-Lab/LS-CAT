#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
//blockDim is num threads/block, multiplied by block number to index to one of them, then select thread inside block via thread Id
int threadID = threadIdx.x + blockIdx.x * blockDim.x;
//Max 65 535 blocks, with 512 threads each ~ 8 million elements, if vector exceeds that amount require a soln
//Run arbitrary number of blocks and threads
//Done at each parallel process, allows a single launch of threads to iteratively cycle through all available indices of vector
//As long as each thread begins at a unique index-val, all will iterate arr without affecting one another
while (threadID < N)
{
c[threadID] = a[threadID] + b[threadID];
//Add
threadID += blockDim.x * gridDim.x;
}
}