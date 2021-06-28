#include "includes.h"
__global__ void gCopyCols(float* out, const float* in, size_t rows, size_t colsIn, const size_t* sourceColIdx, size_t colsOut) {
for(int bid = 0; bid < rows; bid += gridDim.x) {
int j = bid + blockIdx.x;
if(j < rows) {
const float* rowIn = in + j * colsIn;
float* rowOut = out + j * colsOut;

for(int tid = 0; tid < colsOut; tid += blockDim.x) {
int i = tid + threadIdx.x;
if(i < colsOut)
rowOut[i] = rowIn[sourceColIdx[i]];
}
}
}
}