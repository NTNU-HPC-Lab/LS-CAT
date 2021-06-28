#include "includes.h"
__global__ static void resizeBatchKernel(const uint16_t *p_Src, int nSrcPitch, int nSrcHeight, uint8_t *dpDst, int nDstWidth, int nDstHeight, int *lookupTable_cuda) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int tidd = blockIdx.y * blockDim.y + threadIdx.y;
uint4 pF;
int scale = nSrcHeight / nDstHeight;
if (scale == 4) {
uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
int nDstH = nDstHeight;
int nDstW = nDstWidth / 6;
if (tid < nDstW && tidd < nDstH) {
int j = tidd * nSrcPitch * scale;
int k = tid * 32;

pF.x = (uint32_t)p_Src[j + k + 0] + ((uint32_t)p_Src[j + k + 1] << 16);
pF.w = (uint32_t)p_Src[j + k + 6];

v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
u0 = (uint32_t)(pF.x & 0x000003FF);
y1 = (uint32_t)(pF.w & 0x000003FF);

pF.y = (uint32_t)p_Src[j + k + 10] + ((uint32_t)p_Src[j + k + 11] << 16);
pF.z = (uint32_t)p_Src[j + k + 12];

y2 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
u1 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
v1 = (uint32_t)(pF.z & 0x000003FF);

pF.x = (uint32_t)p_Src[j + k + 16] + ((uint32_t)p_Src[j + k + 17] << 16);
pF.z = ((uint32_t)p_Src[j + k + 21] << 16);
pF.w = (uint32_t)p_Src[j + k + 22] + ((uint32_t)p_Src[j + k + 23] << 16);

y3 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
u2 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
v2 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
y4 = (uint32_t)(pF.w & 0x000003FF);

pF.y = ((uint32_t)p_Src[j + k + 27] << 16);

y5 = (uint32_t)((pF.y & 0x3FF00000) >> 20);

k = tid * 6;
j = tidd * nDstWidth;
dpDst[j + k + 0] = lookupTable_cuda[y0];
dpDst[j + k + 1] = lookupTable_cuda[y1];
dpDst[j + k + 2] = lookupTable_cuda[y2];
dpDst[j + k + 3] = lookupTable_cuda[y3];
dpDst[j + k + 4] = lookupTable_cuda[y4];
dpDst[j + k + 5] = lookupTable_cuda[y5];
k = tid * 3;
j = tidd * nDstWidth / 2;
dpDst[j + k + 0] = lookupTable_cuda[u0];
dpDst[j + k + 1] = lookupTable_cuda[u1];
dpDst[j + k + 2] = lookupTable_cuda[u2];
j = tidd * nDstWidth / 2 + nDstWidth * nDstHeight * 3 / 2;
dpDst[j + k + 0] = lookupTable_cuda[v0];
dpDst[j + k + 1] = lookupTable_cuda[v1];
dpDst[j + k + 2] = lookupTable_cuda[v2];
}
}
else if (scale == 6) {
uint32_t v0, y0, u0, y1;
int nDstH = nDstHeight;
int nDstW = nDstWidth / 2;
if (tid < nDstW && tidd < nDstH) {
int j = tidd * nSrcPitch * scale;
int k = tid * 16;
pF.x = (uint32_t)p_Src[j + k + 0] + ((uint32_t)p_Src[j + k + 1] << 16);

v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
u0 = (uint32_t)(pF.x & 0x000003FF);

pF.x = (uint32_t)p_Src[j + k + 8] + ((uint32_t)p_Src[j + k + 9] << 16);

y1 = (uint32_t)((pF.x & 0x000FFC00) >> 10);

k = tid * 2;
j = tidd * nDstWidth;
dpDst[j + k + 0] = lookupTable_cuda[y0];
dpDst[j + k + 1] = lookupTable_cuda[y1];
k = tid;
j = tidd * nDstWidth / 2;
dpDst[j + k + 0] = lookupTable_cuda[u0];
j = tidd * nDstWidth / 2 + nDstWidth * nDstHeight * 3 / 2;
dpDst[j + k + 1] = lookupTable_cuda[v0];
}
}
else if (scale == 2) {
uint32_t v0, y0, u0, y2, u1, y1, u2, y3, v1, y5, v2, y4;
int nDstH = nDstHeight;
int nDstW = nDstWidth / 6;
if (tid < nDstW && tidd < nDstH) {
int j = tidd * nSrcPitch * scale;
int k = tid * 16;
pF.x = (uint32_t)p_Src[j + k + 0] + ((uint32_t)p_Src[j + k + 1] << 16);
pF.y = ((uint32_t)p_Src[j + k + 3] << 16);
pF.z = ((uint32_t)p_Src[j + k + 5] << 16);
pF.w = (uint32_t)p_Src[j + k + 6] + ((uint32_t)p_Src[j + k + 7] << 16);

v0 = (uint32_t)((pF.x & 0x3FF00000) >> 20);
y0 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
u0 = (uint32_t)(pF.x & 0x000003FF);
y1 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
u1 = (uint32_t)((pF.z & 0x3FF00000) >> 20);
v1 = (uint32_t)((pF.w & 0x000FFC00) >> 10);
y2 = (uint32_t)(pF.w & 0x000003FF);

pF.x = (uint32_t)p_Src[j + k + 8] + ((uint32_t)p_Src[j + k + 9] << 16);
pF.y = (uint32_t)p_Src[j + k + 10] + ((uint32_t)p_Src[j + k + 11] << 16);
pF.z = (uint32_t)p_Src[j + k + 12];
pF.w = (uint32_t)p_Src[j + k + 14];

y3 = (uint32_t)((pF.x & 0x000FFC00) >> 10);
y4 = (uint32_t)((pF.y & 0x3FF00000) >> 20);
u2 = (uint32_t)((pF.y & 0x000FFC00) >> 10);
v2 = (uint32_t)(pF.z & 0x000003FF);
y5 = (uint32_t)(pF.w & 0x000003FF);

k = tid * 6;
j = tidd * nDstWidth;
dpDst[j + k + 0] = lookupTable_cuda[y0];
dpDst[j + k + 1] = lookupTable_cuda[y1];
dpDst[j + k + 2] = lookupTable_cuda[y2];
dpDst[j + k + 3] = lookupTable_cuda[y3];
dpDst[j + k + 4] = lookupTable_cuda[y4];
dpDst[j + k + 5] = lookupTable_cuda[y5];
k = tid * 3;
j = tidd * nDstWidth / 2;
dpDst[j + k + 0] = lookupTable_cuda[u0];
dpDst[j + k + 1] = lookupTable_cuda[u1];
dpDst[j + k + 2] = lookupTable_cuda[u2];
j = tidd * nDstWidth / 2 + nDstWidth * nDstHeight * 3 / 2;
dpDst[j + k + 0] = lookupTable_cuda[v0];
dpDst[j + k + 1] = lookupTable_cuda[v1];
dpDst[j + k + 2] = lookupTable_cuda[v2];
}
}
}