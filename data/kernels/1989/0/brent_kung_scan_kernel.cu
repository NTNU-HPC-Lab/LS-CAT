#include "includes.h"


constexpr const int SECTION_SIZE = 2048;
constexpr const int MAX_SECTIONS = 1024;

__device__ void brent_kung_scan_(float *X, float *Y, int InputSize) {
const int bx = blockIdx.x;
const int tx = threadIdx.x;
const int bdx = blockDim.x;

__shared__ float XY[SECTION_SIZE];
int i = 2 * bx * bdx + tx;
if (i < InputSize)
XY[tx] = X[i];
if (i + bdx < InputSize)
XY[tx + bdx] = X[i + bdx];
for (unsigned int stride = 1; stride <= bdx; stride *= 2) {
__syncthreads();
int index = (tx + 1) * 2 * stride - 1;
if (index < SECTION_SIZE) {
XY[index] += XY[index - stride];
}
}
for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
__syncthreads();
int index = (tx + 1) * stride * 2 - 1;
if (index + stride < SECTION_SIZE) {
XY[index + stride] += XY[index];
}
}
__syncthreads();
if (i < InputSize)
Y[i] = XY[tx];
if (i + bdx < InputSize)
Y[i + bdx] = XY[tx + bdx];
}
__global__ void brent_kung_scan_kernel(float *X, float *Y, int InputSize) {
brent_kung_scan_(X, Y, InputSize);
}