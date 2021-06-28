#include "includes.h"
__global__ void yMinDeltaIntegralReplicateFracKernel( const float *intData, float *tmpArray, const int nWindows, const int h, const int w, const float *xMin, const float *xMax, const float *yMin, const float *inData, const int inDataStrideRow, const int strideH, const int strideW) {

// TODO: use block dim instead
const int hOut = (h + strideH - 1) / strideH;
const int wOut = (w + strideW - 1) / strideW;

int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
const int yOut = id % wOut; id /= wOut; // 0-indexed
const int xOut = id % hOut; id /= hOut; // 0-indexed
const int & windowIdx = id;

if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

const int x = xOut*strideH + 1;
const int y = yOut*strideW + 1;

tmpArray += windowIdx * hOut * wOut;

const int xMinInt = (int)ceil(xMin[windowIdx]-1);
const float xMinFrac = xMinInt-xMin[windowIdx]+1;

const int yMinInt = (int)ceil(yMin[windowIdx]-1);
// const float yMinFrac = yMinInt-yMin[windowIdx]+1;

const int xMaxInt = (int)floor(xMax[windowIdx]);
const float xMaxFrac = xMax[windowIdx]-xMaxInt;

// const int yMaxInt = (int)floor(yMax[windowIdx]);
// const float yMaxFrac = yMax[windowIdx]-yMaxInt;

const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
inData[
max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
max(0,min(w-1,y+yMinInt-1))];
const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
inData[
max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
max(0,min(w-1,y+yMinInt-1))];
// const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
//                      inData[
//                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
//                         max(0,min(w-1,y+yMaxInt  ))];
// const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
//                     inData[
//                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
//                         max(0,min(w-1,y+yMaxInt  ))];

float delta = 0;

delta += tlCorner * (x+xMinInt >= h ? 1.0f : xMinFrac);
delta += blCorner * (x+xMaxInt <  1 ? 1.0f : xMaxFrac);

delta +=
intData[max(0,min(x+xMaxInt, h))*(w+1)
+ max(0,min(y+yMinInt  , w))];
delta -=
intData[max(0,min(x+xMaxInt, h))*(w+1)
+ max(0,min(y+yMinInt-1, w))];
delta -=
intData[max(0,min(x+xMinInt, h))*(w+1)
+ max(0,min(y+yMinInt  , w))];
delta +=
intData[max(0,min(x+xMinInt, h))*(w+1)
+ max(0,min(y+yMinInt-1, w))];

delta *= (y+yMinInt >= 1 and y+yMinInt < w);
tmpArray[xOut*wOut + yOut] *= -delta;
}
}