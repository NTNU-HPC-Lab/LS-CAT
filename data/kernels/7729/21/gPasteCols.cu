#include "includes.h"
__global__ void gPasteCols(float* out, const float* in, size_t rows, size_t colsOut, const size_t* targetColIdx, size_t colsIn) {
for(int bid = 0; bid < rows; bid += gridDim.x) {
int j = bid + blockIdx.x;
if(j < rows) {
const float* rowIn = in + j * colsIn;
float* rowOut = out + j * colsOut;

for(int tid = 0; tid < colsIn; tid += blockDim.x) {
int i = tid + threadIdx.x;
if(i < colsIn)
rowOut[targetColIdx[i]] += rowIn[i];
}
}
}
}