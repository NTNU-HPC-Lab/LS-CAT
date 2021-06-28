#include "includes.h"
__global__ void updateGradInputVarScaleKernel( float *gradOutputIntData, float *gradInputData, int h, int w, int nWindows, float *xMin, float *xMax, float *yMin, float *yMax) {

const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

if (x < h and y < w) {

int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;
double outValue = 0;

for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

xMinCurr = (int)ceil(-xMax[windowIdx]);
yMinCurr = (int)ceil(-yMax[windowIdx]);

xMaxCurr = (int)floor(-xMin[windowIdx]) + 1;
yMaxCurr = (int)floor(-yMin[windowIdx]) + 1;

// The following code block implements these lines
// as if they were executed simultaneously (see `void updateGradInputFrac()`):
// xMinCurr = (x == 0   and xMaxCurr >= 0 ? 0    : xMinCurr);
// xMaxCurr = (x == h-1 and xMinCurr <= 0 ? h+66 : xMaxCurr);
// yMinCurr = (y == 0   and yMaxCurr >= 0 ? 0    : yMinCurr);
// yMaxCurr = (y == w-1 and yMinCurr <= 0 ? w+66 : yMaxCurr);

bool needToChangeMin, needToChangeMax;

needToChangeMin = x == 0   and xMaxCurr >= 0;
needToChangeMax = x == h-1 and xMinCurr <= 0;
if (needToChangeMin) xMinCurr = 0;
if (needToChangeMax) xMaxCurr = h+66;

needToChangeMin = y == 0   and yMaxCurr >= 0;
needToChangeMax = y == w-1 and yMinCurr <= 0;
if (needToChangeMin) yMinCurr = 0;
if (needToChangeMax) yMaxCurr = w+66;

const int t = max(0, min(x+xMinCurr, h) );
const int b = max(0, min(x+xMaxCurr, h) );
const int l = max(0, min(y+yMinCurr, w) );
const int r = max(0, min(y+yMaxCurr, w) );

outValue += gradOutputIntData[b*(w+1) + r];
outValue -= gradOutputIntData[t*(w+1) + r];
outValue -= gradOutputIntData[b*(w+1) + l];
outValue += gradOutputIntData[t*(w+1) + l];

// go to the next channel
gradOutputIntData += (h+1)*(w+1);
}

gradInputData[x*w + y] = outValue;
}
}