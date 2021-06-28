#include "includes.h"



__global__ void bitonicSort2(int *inputArray, const unsigned int stage, const unsigned int passOfStage, const unsigned int width) {
int4 *theArray = (int4 *)inputArray;
size_t i = blockIdx.x * blockDim.x + threadIdx.x; // get_global_id(0);
int4 srcLeft, srcRight, mask;
int4 imask10 = make_int4(0, 0, -1, -1);
int4 imask11 = make_int4(0, -1, 0, -1);
const unsigned int dir = 0;
if (stage > 0) {
if (passOfStage > 0) // upper level pass, exchange between two fours
{
size_t r = 1 << (passOfStage - 1);
size_t lmask = r - 1;
size_t left = ((i >> (passOfStage - 1)) << passOfStage) + (i & lmask);
size_t right = left + r;

srcLeft = theArray[left];
srcRight = theArray[right];

// mask = srcLeft < srcRight;
mask.x = srcLeft.x < srcRight.x;
mask.y = srcLeft.y < srcRight.y;
mask.z = srcLeft.z < srcRight.z;
mask.w = srcLeft.w < srcRight.w;

// int4 imin = (srcLeft & mask) | (srcRight & ~mask);
int4 imin;
imin.x = (srcLeft.x & mask.x) | (srcRight.x & ~mask.x);
imin.y = (srcLeft.y & mask.y) | (srcRight.y & ~mask.y);
imin.z = (srcLeft.z & mask.z) | (srcRight.z & ~mask.z);
imin.w = (srcLeft.w & mask.w) | (srcRight.w & ~mask.w);

// int4 imax = (srcLeft & ~mask) | (srcRight & mask);
int4 imax;
imax.x = (srcLeft.x & ~mask.x) | (srcRight.x & mask.x);
imax.y = (srcLeft.y & ~mask.y) | (srcRight.y & mask.y);
imax.z = (srcLeft.z & ~mask.z) | (srcRight.z & mask.z);
imax.w = (srcLeft.w & ~mask.w) | (srcRight.w & mask.w);

if (((i >> (stage - 1)) & 1) ^ dir) {
theArray[left] = imin;
theArray[right] = imax;
} else {
theArray[right] = imin;
theArray[left] = imax;
}
} else // last pass, sort inside one four
{
srcLeft = theArray[i];
// srcRight = srcLeft.zwxy;
srcRight = make_int4(srcLeft.z, srcLeft.w, srcLeft.x, srcLeft.y);

// mask = (srcLeft < srcRight) ^ imask10;
mask.x = (srcLeft.x < srcRight.x) ^ imask10.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask10.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask10.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask10.w;

if (((i >> stage) & 1) ^ dir) {
// srcLeft = (srcLeft & mask) | (srcRight & ~mask);
srcLeft.x = (srcLeft.x & mask.x) | (srcRight.x & ~mask.x);
srcLeft.y = (srcLeft.y & mask.y) | (srcRight.y & ~mask.y);
srcLeft.z = (srcLeft.z & mask.z) | (srcRight.z & ~mask.z);
srcLeft.w = (srcLeft.w & mask.w) | (srcRight.w & ~mask.w);

// srcRight = srcLeft.yxwz;
srcRight = make_int4(srcLeft.y, srcLeft.x, srcLeft.w, srcLeft.z);

// mask = (srcLeft < srcRight) ^ imask11;
mask.x = (srcLeft.x < srcRight.x) ^ imask11.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask11.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask11.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask11.w;

// theArray[i] = (srcLeft & mask) | (srcRight & ~mask);
theArray[i].x = (srcLeft.x & mask.x) | (srcRight.x & ~mask.x);
theArray[i].y = (srcLeft.y & mask.y) | (srcRight.y & ~mask.y);
theArray[i].z = (srcLeft.z & mask.z) | (srcRight.z & ~mask.z);
theArray[i].w = (srcLeft.w & mask.w) | (srcRight.w & ~mask.w);
} else {
// srcLeft = (srcLeft & ~mask) | (srcRight & mask);
srcLeft.x = (srcLeft.x & ~mask.x) | (srcRight.x & mask.x);
srcLeft.y = (srcLeft.y & ~mask.y) | (srcRight.y & mask.y);
srcLeft.z = (srcLeft.z & ~mask.z) | (srcRight.z & mask.z);
srcLeft.w = (srcLeft.w & ~mask.w) | (srcRight.w & mask.w);

// srcRight = srcLeft.yxwz;
srcRight = make_int4(srcLeft.y, srcLeft.x, srcLeft.w, srcLeft.z);

// mask = (srcLeft < srcRight) ^ imask11;
mask.x = (srcLeft.x < srcRight.x) ^ imask11.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask11.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask11.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask11.w;

// theArray[i] = (srcLeft & ~mask) | (srcRight & mask);
theArray[i].x = (srcLeft.x & ~mask.x) | (srcRight.x & mask.x);
theArray[i].y = (srcLeft.y & ~mask.y) | (srcRight.y & mask.y);
theArray[i].z = (srcLeft.z & ~mask.z) | (srcRight.z & mask.z);
theArray[i].w = (srcLeft.w & ~mask.w) | (srcRight.w & mask.w);
}
}
} else // first stage, sort inside one four
{
int4 imask0 = make_int4(0, -1, -1, 0);
srcLeft = theArray[i];

// srcRight = srcLeft.yxwz;
srcRight = make_int4(srcLeft.y, srcLeft.x, srcLeft.w, srcLeft.z);

// mask = (srcLeft < srcRight) ^ imask0;
mask.x = (srcLeft.x < srcRight.x) ^ imask0.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask0.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask0.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask0.w;

if (dir) {
// srcLeft = (srcLeft & mask) | (srcRight & ~mask);
srcLeft.x = (srcLeft.x & mask.x) | (srcRight.x & ~mask.x);
srcLeft.y = (srcLeft.y & mask.y) | (srcRight.y & ~mask.y);
srcLeft.z = (srcLeft.z & mask.z) | (srcRight.z & ~mask.z);
srcLeft.w = (srcLeft.w & mask.w) | (srcRight.w & ~mask.w);
} else {
// srcLeft = (srcLeft & ~mask) | (srcRight & mask);
srcLeft.x = (srcLeft.x & ~mask.x) | (srcRight.x & mask.x);
srcLeft.y = (srcLeft.y & ~mask.y) | (srcRight.y & mask.y);
srcLeft.z = (srcLeft.z & ~mask.z) | (srcRight.z & mask.z);
srcLeft.w = (srcLeft.w & ~mask.w) | (srcRight.w & mask.w);
}

// srcRight = srcLeft.zwxy;
srcRight = make_int4(srcLeft.z, srcLeft.w, srcLeft.x, srcLeft.y);

// mask = (srcLeft < srcRight) ^ imask10;
mask.x = (srcLeft.x < srcRight.x) ^ imask10.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask10.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask10.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask10.w;

if ((i & 1) ^ dir) {
// srcLeft = (srcLeft & mask) | (srcRight & ~mask);
srcLeft.x = (srcLeft.x & mask.x) | (srcRight.x & ~mask.x);
srcLeft.y = (srcLeft.y & mask.y) | (srcRight.y & ~mask.y);
srcLeft.z = (srcLeft.z & mask.z) | (srcRight.z & ~mask.z);
srcLeft.w = (srcLeft.w & mask.w) | (srcRight.w & ~mask.w);

// srcRight = srcLeft.yxwz;
srcRight = make_int4(srcLeft.y, srcLeft.x, srcLeft.w, srcLeft.z);

// mask = (srcLeft < srcRight) ^ imask11;
mask.x = (srcLeft.x < srcRight.x) ^ imask11.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask11.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask11.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask11.w;

// theArray[i] = (srcLeft & mask) | (srcRight & ~mask);
theArray[i].x = (srcLeft.x & mask.x) | (srcRight.x & ~mask.x);
theArray[i].y = (srcLeft.y & mask.y) | (srcRight.y & ~mask.y);
theArray[i].z = (srcLeft.z & mask.z) | (srcRight.z & ~mask.z);
theArray[i].w = (srcLeft.w & mask.w) | (srcRight.w & ~mask.w);
} else {
// srcLeft = (srcLeft & ~mask) | (srcRight & mask);
srcLeft.x = (srcLeft.x & ~mask.x) | (srcRight.x & mask.x);
srcLeft.y = (srcLeft.y & ~mask.y) | (srcRight.y & mask.y);
srcLeft.z = (srcLeft.z & ~mask.z) | (srcRight.z & mask.z);
srcLeft.w = (srcLeft.w & ~mask.w) | (srcRight.w & mask.w);

// srcRight = srcLeft.yxwz;
srcRight = make_int4(srcLeft.y, srcLeft.x, srcLeft.w, srcLeft.z);

// mask = (srcLeft < srcRight) ^ imask11;
mask.x = (srcLeft.x < srcRight.x) ^ imask11.x;
mask.y = (srcLeft.y < srcRight.y) ^ imask11.y;
mask.z = (srcLeft.z < srcRight.z) ^ imask11.z;
mask.w = (srcLeft.w < srcRight.w) ^ imask11.w;

// theArray[i] = (srcLeft & ~mask) | (srcRight & mask);
theArray[i].x = (srcLeft.x & ~mask.x) | (srcRight.x & mask.x);
theArray[i].y = (srcLeft.y & ~mask.y) | (srcRight.y & mask.y);
theArray[i].z = (srcLeft.z & ~mask.z) | (srcRight.z & mask.z);
theArray[i].w = (srcLeft.w & ~mask.w) | (srcRight.w & mask.w);
}
}
}