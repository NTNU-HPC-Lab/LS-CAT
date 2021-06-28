#include "includes.h"
__global__ void nan_kernel(float* data, const bool* mask, int len, float nan) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid >= len) return;
if (!mask[tid]) data[tid] = nan;
}