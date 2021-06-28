#include "includes.h"
__global__ void kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows){
__shared__ int sourceRowIndices[32];
const int startTargetRowI = blockIdx.x * 32;
const int tid = threadIdx.x;
const int localNRowIs = min(32, nRowIs-startTargetRowI);

// cooperatively load 32 row indices
if (tid < localNRowIs){
sourceRowIndices[tid] = int(indices[startTargetRowI + tid]);
if (sourceRowIndices[tid]<0)
sourceRowIndices[tid] += nSourceRows;
if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nSourceRows)
sourceRowIndices[tid] = -1;
}
__syncthreads();

// copy 32 rows
for (int i=0; i<localNRowIs; i++){
const int targetRowI = startTargetRowI + i, sourceRowI = sourceRowIndices[i];
for (int colI=tid; colI<nCols; colI+=32)
target[targetRowI * nCols + colI] = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
}
}