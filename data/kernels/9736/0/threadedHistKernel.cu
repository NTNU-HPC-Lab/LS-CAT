#include "includes.h"


__global__ void threadedHistKernel(int *threadedHist, int *arr, const int blockSize, const int valRange, const int threadBlockSize)
{
int val,
bid = blockIdx.x,
tid = threadIdx.x,
pid = bid*blockSize + tid;  //positional ID

// each thread takes info from its given info and increases the relevant position on the threadedHist
for (int i = 0; i < threadBlockSize; i++)
{
val = arr[pid*threadBlockSize + i];
threadedHist[valRange*pid + val]++;

}
}