#include "includes.h"
__global__ void kSwapRows(float* source, float* target, float* indices1, float* indices2, int nRowIs, int nCols, int nRows){
__shared__ int sourceRowIndices[32], targetRowIndices[32];
const int startRowI = blockIdx.x * 32;
const int tid = threadIdx.x;
const int localNRowIs = min(32, nRowIs-startRowI);

// cooperatively load 32 row indices
if (tid < localNRowIs){
sourceRowIndices[tid] = int(indices1[startRowI + tid]);
targetRowIndices[tid] = int(indices2[startRowI + tid]);
if (sourceRowIndices[tid]<0)
sourceRowIndices[tid] += nRows;
if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nRows)
sourceRowIndices[tid] = -1;
if (targetRowIndices[tid]<0)
targetRowIndices[tid] += nRows;
if (targetRowIndices[tid]<0 || targetRowIndices[tid]>=nRows)
targetRowIndices[tid] = -1;
}
__syncthreads();

// copy 32 rows
for (int i=0; i<localNRowIs; i++){
const int sourceRowI = sourceRowIndices[i], targetRowI = targetRowIndices[i];
for (int colI=tid; colI<nCols; colI+=32) {
const float temp1 = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
const float temp2 = targetRowI==-1 ? (1.0/0.0 -1.0/0.0) : target[targetRowI * nCols + colI];
if (sourceRowI != -1)
source[sourceRowI * nCols + colI] =  temp2;
if (targetRowI != -1)
target[targetRowI * nCols + colI] = temp1;
}
}
}