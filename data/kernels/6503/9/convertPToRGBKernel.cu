#include "includes.h"
__global__ static void convertPToRGBKernel(const uint16_t *dpSrc, uint8_t *tt1, int nSrcWidth, int nDstWidth, int nDstHeight, int *lookupTable) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int tidd = blockIdx.y * blockDim.y + threadIdx.y;
uint32_t v0, y0, u0, y1;
uint16_t tt[6];
int nDstH = nDstHeight;
int nDstW = nSrcWidth / 2;
if (tid < nDstW && tidd < nDstH) {
int k = tid * 2;
int j = tidd * nSrcWidth;
y0 = (uint32_t)dpSrc[j + k + 0] * 1000;
y1 = (uint32_t)dpSrc[j + k + 1] * 1000;
k = tid;
j = tidd * nSrcWidth / 2 + nDstHeight * nSrcWidth;
u0 = (uint32_t)dpSrc[j + k + 0];
j = tidd * nSrcWidth / 2 + nDstHeight * nSrcWidth * 3 / 2;
v0 = (uint32_t)dpSrc[j + k + 0];

k = tid * 6;
j = tidd * nDstWidth * 3;
int r = 1407 * v0 - 720384, g = 716 * v0 + 345 * u0 - 543232, b = 1779 * u0 - 910848;
tt[0] = (y0 + r) / 1000;
tt[1] = (y0 - g) / 1000;
tt[2] = (y0 + b) / 1000;
tt[3] = (y1 + r) / 1000;
tt[4] = (y1 - g) / 1000;
tt[5] = (y1 + b) / 1000;

tt1[j + k + 0] = lookupTable[tt[0]];
tt1[j + k + 1] = lookupTable[tt[1]];
tt1[j + k + 2] = lookupTable[tt[2]];
tt1[j + k + 3] = lookupTable[tt[3]];
tt1[j + k + 4] = lookupTable[tt[4]];
tt1[j + k + 5] = lookupTable[tt[5]];
}
}