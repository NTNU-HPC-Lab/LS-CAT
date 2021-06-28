#include "includes.h"
__global__ void kSetSelectedRows(float* target, float* source, float* indices, int nRowIs, int nCols, int nTargetRows){
__shared__ int targetRowIndices[32];
const int startSourceRowI = blockIdx.x * 32;
const int tid = threadIdx.x;
const int localNRowIs = min(32, nRowIs-startSourceRowI);

// cooperatively load 32 row indices
if (tid < localNRowIs){
targetRowIndices[tid] = int(indices[startSourceRowI + tid]);
if (targetRowIndices[tid]<0)
targetRowIndices[tid] += nTargetRows;
if (targetRowIndices[tid]<0 || targetRowIndices[tid]>=nTargetRows)
targetRowIndices[tid] = -1;
}
__syncthreads();

// copy 32 rows
for (int i=0; i<localNRowIs; i++){
const int sourceRowI = startSourceRowI + i, targetRowI = targetRowIndices[i];
for (int colI=tid; colI<nCols; colI+=32)
target[targetRowI * nCols + colI] = targetRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
}
}