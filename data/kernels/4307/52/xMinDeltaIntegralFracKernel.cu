#include "includes.h"
__global__ void xMinDeltaIntegralFracKernel( const float *intData, float *tmpArray, const int nWindows, const int h, const int w, const float *xMin, const float *yMin, const float *yMax, const float *inData, const int inDataStrideRow) {

int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
const int y = id % w + 1; id /= w; // 1-indexed
const int x = id % h + 1; id /= h; // 1-indexed
const int & windowIdx = id;

if (windowIdx < nWindows and x <= h and y <= w) {

tmpArray += windowIdx * h * w;

const int rem = windowIdx % 4;

if (rem == 0) {
tmpArray[(x-1)*w + (y-1)] = 0;
} else {

const float xMinStretched = rem == 0 ? -h :
xMin[3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
// const float xMaxStretched = rem == 1 ?  h :
//     xMax[3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
const float yMinStretched = rem == 2 ? -w :
yMin[3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
const float yMaxStretched = rem == 3 ?  w :
yMax[3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

const int xMinInt = (int)ceil(xMinStretched-1);
// const float xMinFrac = xMinInt-xMinStretched+1;

const int yMinInt = (int)ceil(yMinStretched-1);
const float yMinFrac = yMinInt-yMinStretched+1;

// const int xMaxInt = (int)floor(xMaxStretched);
// const float xMaxFrac = xMaxStretched-xMaxInt;

const int yMaxInt = (int)floor(yMaxStretched);
const float yMaxFrac = yMaxStretched-yMaxInt;

const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
inData[
max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
max(0,min(w-1,y+yMinInt-1))];
// const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
//                     inData[
//                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
//                         max(0,min(w-1,y+yMinInt-1))];
const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
inData[
max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
max(0,min(w-1,y+yMaxInt  ))];
// const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
//                     inData[
//                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
//                         max(0,min(w-1,y+yMaxInt  ))];

float delta = 0;

delta += trCorner * (y+yMaxInt <  1 ? 1.0f : yMaxFrac);
delta += tlCorner * (y+yMinInt >= w ? 1.0f : yMinFrac);

delta +=
intData[max(0,min(x+xMinInt  , h))*(w+1)
+ max(0,min(y+yMaxInt, w))];
delta -=
intData[max(0,min(x+xMinInt-1, h))*(w+1)
+ max(0,min(y+yMaxInt, w))];
delta -=
intData[max(0,min(x+xMinInt  , h))*(w+1)
+ max(0,min(y+yMinInt, w))];
delta +=
intData[max(0,min(x+xMinInt-1, h))*(w+1)
+ max(0,min(y+yMinInt, w))];

delta *= (x+xMinInt >= 1 and x+xMinInt < h);
tmpArray[(x-1)*w + (y-1)] *= -delta;
}
}
}