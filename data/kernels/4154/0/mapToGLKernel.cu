#include "includes.h"


__global__ static void mapToGLKernel(uint8_t *dSrc, uint8_t *dDst, int nWidth, int nHeight) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int tidd = blockIdx.y * blockDim.y + threadIdx.y;
if (tid < nWidth && tidd < nHeight) {
int j = tidd * nWidth * 3;
int k = tid * 3;
dDst[j + k + 0] = dSrc[j + k + 0];
dDst[j + k + 1] = dSrc[j + k + 1];
dDst[j + k + 2] = dSrc[j + k + 2];
}
}