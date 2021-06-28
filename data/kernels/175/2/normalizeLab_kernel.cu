#include "includes.h"
// vim: ts=4 syntax=cpp comments=


#define XBLOCK 16
#define YBLOCK 16





__global__ void normalizeLab_kernel(uint width, uint height, float* devL, float* devA, float* devB) {
int x0 = blockDim.x * blockIdx.x + threadIdx.x;
int y0 = blockDim.y * blockIdx.y + threadIdx.y;
if ((x0 < width) && (y0 < height)) {
int index = y0 * width + x0;
const float ab_min = -73;
const float ab_max = 95;
const float ab_range = ab_max - ab_min;
/* normalize Lab image */
float l_val = devL[index] / 100.0f;
float a_val = (devA[index] - ab_min) / ab_range;
float b_val = (devB[index] - ab_min) / ab_range;
if (l_val < 0) { l_val = 0; } else if (l_val > 1) { l_val = 1; }
if (a_val < 0) { a_val = 0; } else if (a_val > 1) { a_val = 1; }
if (b_val < 0) { b_val = 0; } else if (b_val > 1) { b_val = 1; }
devL[index] = l_val;
devA[index] = a_val;
devB[index] = b_val;
}
}