#include "includes.h"
__global__ static void convertVToRGBKernel(const uint16_t *pV210, uint8_t *tt1, int nSrcWidth, int nDstWidth, int nDstHeight, int *lookupTable) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int tidd = blockIdx.y * blockDim.y + threadIdx.y;
uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
uint16_t tt[6];
uint4 pF;
int nDstH = nDstHeight;
int nDstW = nSrcWidth / 8;

if (tid < nDstW && tidd < nDstH) {
int j = tidd * nSrcWidth;
int k = tid * 8;
pF.x = (uint32_t)pV210[j + k + 0] + ((uint32_t)pV210[j + k + 1] << 16);
pF.y = (uint32_t)pV210[j + k + 2] + ((uint32_t)pV210[j + k + 3] << 16);
pF.z = (uint32_t)pV210[j + k + 4] + ((uint32_t)pV210[j + k + 5] << 16);
pF.w = (uint32_t)pV210[j + k + 6] + ((uint32_t)pV210[j + k + 7] << 16);

v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10) * 1000;
u0 = (uint32_t)(pF.x & 0x000003FF);
y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20) * 1000;
u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
y1 = (uint32_t)(pF.y & 0x000003FF) * 1000;
u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
y3 = (uint32_t)((pF.z & 0x000FFC00) >> 10) * 1000;
v1 = (uint32_t)(pF.z & 0x000003FF);
y5 = (uint32_t)((pF.w & 0x3FF00000) >> 20) * 1000;
v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
y4 = (uint32_t)(pF.w & 0x000003FF) * 1000;

k = tid * 18;
j *= 9;
j /= 4;
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

r = 1407 * v1 - 720384, g = 716 * v1 + 345 * u1 - 543232, b = 1779 * u1 - 910848;
tt[0] = (y2 + r) / 1000;
tt[1] = (y2 - g) / 1000;
tt[2] = (y2 + b) / 1000;

tt[3] = (y3 + r) / 1000;
tt[4] = (y3 - g) / 1000;
tt[5] = (y3 + b) / 1000;

tt1[j + k + 6] = lookupTable[tt[0]];
tt1[j + k + 7] = lookupTable[tt[1]];
tt1[j + k + 8] = lookupTable[tt[2]];

tt1[j + k + 9] = lookupTable[tt[3]];
tt1[j + k + 10] = lookupTable[tt[4]];
tt1[j + k + 11] = lookupTable[tt[5]];

r = 1407 * v2 - 720384, g = 716 * v2 + 345 * u2 - 543232, b = 1779 * u2 - 910848;
tt[0] = (y4 + r) / 1000;
tt[1] = (y4 - g) / 1000;
tt[2] = (y4 + b) / 1000;

tt[3] = (y5 + r) / 1000;
tt[4] = (y5 - g) / 1000;
tt[5] = (y5 + b) / 1000;

tt1[j + k + 12] = lookupTable[tt[0]];
tt1[j + k + 13] = lookupTable[tt[1]];
tt1[j + k + 14] = lookupTable[tt[2]];

tt1[j + k + 15] = lookupTable[tt[3]];
tt1[j + k + 16] = lookupTable[tt[4]];
tt1[j + k + 17] = lookupTable[tt[5]];
}
}