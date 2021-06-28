#include "includes.h"
__global__ void xMinDeltaIntegralKernel( const float *intData, const int intDataStrideChannel, float *tmpArray, const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w, const float *xMin, const float *yMin, const float *yMax) {

int id = NUM_THREADS * blockIdx.x + threadIdx.x;
tmpArray += id; // tmpArray now points to our output pixel

const int y = id % w + 1; id /= w; // 1-indexed
const int x = id % h + 1; id /= h; // 1-indexed
const int windowIdx = id % nWindows; id /= nWindows;

// `id` is now is now the current global input plane number
intData  += id * intDataStrideChannel;

const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
const int & batchIdx = id;

if (batchIdx < batchSize) {

const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
// const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
const int yMaxInt = (int)floor(yMax[globalWindowIdx]);

float delta = 0;

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
*tmpArray = -delta;
}
}