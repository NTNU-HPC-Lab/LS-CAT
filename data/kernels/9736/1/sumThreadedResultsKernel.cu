#include "includes.h"


__global__ void sumThreadedResultsKernel(long *dev_hist, int *dev_threadedHist, const int valRange, const int Blocks)
{
//e.g. tid from 0 to valRange-1, blocks = THREADS_PER_BLOCK * NO_BLOCKS
int tid = threadIdx.x;

for (int bl = 0; bl < Blocks; bl++)
{
dev_hist[tid] += dev_threadedHist[bl*valRange + tid];
}
}