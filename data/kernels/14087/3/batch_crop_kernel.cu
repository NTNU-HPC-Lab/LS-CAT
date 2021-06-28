#include "includes.h"
__global__ void batch_crop_kernel(float* input, const int nCropRows, const int nCropCols, const int iH, const int iW, const int nPlanes){
const int plane = blockIdx.x;
if (plane >= nPlanes)
return;

input += plane * iH * iW;
const int tx = threadIdx.x;
const int ty = threadIdx.y;

if (ty < iH && (ty > iH-nCropRows-1 || ty < nCropRows)) {
input[ty*iW + tx] = 0;
}
if (tx < iW && (tx > iW-nCropCols-1 || tx < nCropCols)) {
input[ty*iW + tx] = 0;
}
}