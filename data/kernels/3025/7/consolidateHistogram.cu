#include "includes.h"
__global__ void consolidateHistogram(ulong*blockHistograms,ulong* cudaHistogram,uint numBlocks) {
int tid = threadIdx.x;

for (uint j=0;j<numBlocks;j++) {
cudaHistogram[tid]+=blockHistograms[j*256+tid];
}
}