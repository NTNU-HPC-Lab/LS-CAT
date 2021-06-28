#include "includes.h"
__global__ void xMaxDeltaIntegralFracKernel( const float *intData, const int intDataStrideChannel, float *tmpArray, const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w, const float *xMax, const float *yMin, const float *yMax, const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {

int id = NUM_THREADS * blockIdx.x + threadIdx.x;
tmpArray += id; // tmpArray now points to our output pixel

const int y = id % w + 1; id /= w; // 1-indexed
const int x = id % h + 1; id /= h; // 1-indexed
const int windowIdx = id % nWindows; id /= nWindows;

// `id` is now is now the current global input plane number
intData  += id * intDataStrideChannel;
inData   += id *  inDataStrideChannel;

const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
const int & batchIdx = id;

if (batchIdx < batchSize) {

// const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
// const float xMinFrac = xMinInt-xMin[globalWindowIdx]+1;

const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
const float yMinFrac = yMinInt-yMin[globalWindowIdx]+1;

const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
// const float xMaxFrac = xMax[globalWindowIdx]-xMaxInt;

const int yMaxInt = (int)floor(yMax[globalWindowIdx]);
const float yMaxFrac = yMax[globalWindowIdx]-yMaxInt;

int valid;

valid = not (y+yMinInt < 1) & not (y+yMinInt > w) & not (x+xMaxInt >= h);
const float blCorner = valid * inData[
max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
max(0,min(w-1,y+yMinInt-1))];

valid = not (y+yMaxInt < 0) & not (y+yMaxInt >= w) & not (x+xMaxInt >= h);
const float brCorner = valid * inData[
max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
max(0,min(w-1,y+yMaxInt  ))];

float delta = 0;

delta += brCorner * yMaxFrac;
delta += blCorner * yMinFrac;

delta +=
intData[max(0,min(x+xMaxInt+1, h))*(w+1)
+ max(0,min(y+yMaxInt  , w))];
delta -=
intData[max(0,min(x+xMaxInt  , h))*(w+1)
+ max(0,min(y+yMaxInt  , w))];
delta -=
intData[max(0,min(x+xMaxInt+1, h))*(w+1)
+ max(0,min(y+yMinInt  , w))];
delta +=
intData[max(0,min(x+xMaxInt  , h))*(w+1)
+ max(0,min(y+yMinInt  , w))];

delta *= (x+xMaxInt >= 0) & (x+xMaxInt < h);
*tmpArray = delta;
}
}