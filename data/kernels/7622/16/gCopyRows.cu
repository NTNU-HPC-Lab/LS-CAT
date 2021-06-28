#include "includes.h"
__global__ void gCopyRows(float* out, const float* in, size_t cols, const size_t* sourceRowIdx, size_t rows) {
for(int bid = 0; bid < rows; bid += gridDim.x) {
int j = bid + blockIdx.x;
if(j < rows) {
size_t dstId = j;
size_t srcId = sourceRowIdx[j];

float* rowOut = out + dstId * cols;
const float* rowIn = in + srcId * cols;

for(int tid = 0; tid < cols; tid += blockDim.x) {
int i = tid + threadIdx.x;
if(i < cols)
rowOut[i] = rowIn[i];
}
}
}
}