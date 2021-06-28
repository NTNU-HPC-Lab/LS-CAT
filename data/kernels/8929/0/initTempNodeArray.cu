#include "includes.h"
__global__ void initTempNodeArray( const int hitNum, const int allowableGap, int* tempNodeArray_score, int* tempNodeArray_vertical, int* tempNodeArray_horizontal, int* tempNodeArray_matchNum) {
const int bIdx = gridDim.x * blockIdx.y + blockIdx.x;
const int idx  = blockDim.x * bIdx + threadIdx.x;
const int halfTempNodeWidth = allowableGap + MARGIN;
const int tempNodeWidth     = 1 + 2 * halfTempNodeWidth;
if(idx < hitNum * tempNodeWidth) {
const int bandIdx = idx / hitNum;
if(bandIdx < halfTempNodeWidth) {
tempNodeArray_score     [idx] = -30000;
tempNodeArray_vertical  [idx] = -30000;
tempNodeArray_horizontal[idx] = -30000;
tempNodeArray_matchNum  [idx] = -30000;
} else if(bandIdx == halfTempNodeWidth) {
tempNodeArray_score     [idx] = 0;
tempNodeArray_vertical  [idx] = GAP_OPEN_POINT;
tempNodeArray_horizontal[idx] = GAP_OPEN_POINT;
tempNodeArray_matchNum  [idx] = 0;
} else {
const int i = bandIdx - halfTempNodeWidth;
const int tempScore = i * GAP_POINT + GAP_OPEN_POINT;
tempNodeArray_score     [idx] = tempScore;
tempNodeArray_vertical  [idx] = tempScore + GAP_OPEN_POINT;
tempNodeArray_horizontal[idx] = tempScore;
tempNodeArray_matchNum  [idx] = 0;
}
}
}