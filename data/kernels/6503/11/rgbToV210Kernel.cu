#include "includes.h"
__global__ static void rgbToV210Kernel(uint16_t *pSrc, uint16_t *pDst, int nSrcWidth, int nHeight) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int tidd = blockIdx.y * blockDim.y + threadIdx.y;
uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
uint3 rgb;
uint4 pF;
int nDstW = nSrcWidth / 18;
int nDstH = nHeight;
if (tid < nDstW && tidd < nDstH) {
int k = tid * 18;
int j = tidd * nSrcWidth;
rgb.x = pSrc[j + k + 0];
rgb.y = pSrc[j + k + 1];
rgb.z = pSrc[j + k + 2];

y0 = (rgb.x * 299 + rgb.y * 587 + rgb.z * 114) / 1000;
u0 = (512000 - rgb.x * 169 - rgb.y * 332 + rgb.z * 500) / 1000;
v0 = (512000 + rgb.x * 500 - rgb.y * 419 - rgb.z * 81) / 1000;

rgb.x = pSrc[j + k + 3];
rgb.y = pSrc[j + k + 4];
rgb.z = pSrc[j + k + 5];

y1 = (rgb.x * 299 + rgb.y * 587 + rgb.z * 114) / 1000;
u1 = (512000 - rgb.x * 169 - rgb.y * 332 + rgb.z * 500) / 1000;
v1 = (512000 + rgb.x * 500 - rgb.y * 419 - rgb.z * 81) / 1000;

rgb.x = pSrc[j + k + 6];
rgb.y = pSrc[j + k + 7];
rgb.z = pSrc[j + k + 8];

y2 = (rgb.x * 299 + rgb.y * 587 + rgb.z * 114) / 1000;
u2 = (512000 - rgb.x * 169 - rgb.y * 332 + rgb.z * 500) / 1000;
v2 = (512000 + rgb.x * 500 - rgb.y * 419 - rgb.z * 81) / 1000;

rgb.x = pSrc[j + k + 9];
rgb.y = pSrc[j + k + 10];
rgb.z = pSrc[j + k + 11];

y3 = (rgb.x * 299 + rgb.y * 587 + rgb.z * 114) / 1000;

rgb.x = pSrc[j + k + 12];
rgb.y = pSrc[j + k + 13];
rgb.z = pSrc[j + k + 14];

y4 = (rgb.x * 299 + rgb.y * 587 + rgb.z * 114) / 1000;

rgb.x = pSrc[j + k + 15];
rgb.y = pSrc[j + k + 16];
rgb.z = pSrc[j + k + 17];

y5 = (rgb.x * 299 + rgb.y * 587 + rgb.z * 114) / 1000;

pF.x = (v0 << 20) | (y0 << 10) | u0;
pF.y = (y2 << 20) | (u1 << 10) | y1;
pF.z = (u2 << 20) | (y3 << 10) | v1;
pF.w = (y5 << 20) | (v2 << 10) | y4;

k = tid * 8;
j *= 4;
j /= 9;
pDst[j + k + 0] = (uint32_t)(pF.x & 0x0000FFFF);
pDst[j + k + 1] = (uint32_t)(pF.x >> 16);
pDst[j + k + 2] = (uint32_t)(pF.y & 0x0000FFFF);
pDst[j + k + 3] = (uint32_t)(pF.y >> 16);
pDst[j + k + 4] = (uint32_t)(pF.z & 0x0000FFFF);
pDst[j + k + 5] = (uint32_t)(pF.z >> 16);
pDst[j + k + 6] = (uint32_t)(pF.w & 0x0000FFFF);
pDst[j + k + 7] = (uint32_t)(pF.w >> 16);
}
}