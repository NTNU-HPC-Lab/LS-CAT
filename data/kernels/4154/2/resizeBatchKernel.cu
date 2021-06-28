#include "includes.h"
__global__ static void resizeBatchKernel(const uint8_t *p_Src, int nSrcPitch, int nSrcHeight, uint8_t *p_dst, int nDstWidth, int nDstHeight) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int tidd = blockIdx.y * blockDim.y + threadIdx.y;
uchar3 rgb;
int nDstW = nDstWidth;
int nDstH = nDstHeight;
int yScale = nSrcHeight / nDstHeight;
int xScale = 3 * (nSrcPitch / nDstWidth);
if (tid < nDstW && tidd < nDstH) {
int j = tidd * yScale * nSrcPitch * 3;
int k = tid * xScale;
rgb.x = p_Src[j + k + 0];
rgb.y = p_Src[j + k + 1];
rgb.z = p_Src[j + k + 2];
k = tid * 3;
j = tidd * nDstWidth * 3;
p_dst[j + k + 0] = rgb.x;
p_dst[j + k + 1] = rgb.y;
p_dst[j + k + 2] = rgb.z;
}
}