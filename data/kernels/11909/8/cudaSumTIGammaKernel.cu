#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaSumTIGammaKernel(unsigned char *tipX1, double *x2, double *tipVector, double *sumtable, int limit) {
const int n = blockIdx.x * blockDim.x + threadIdx.x;
if (n >= limit) {
return;
}
const int i = n / 4, l = n % 4;
double *left = &(tipVector[4 * tipX1[i]]);
double *right = &(x2[16 * i + l * 4]);
double *sum = &sumtable[i * 16 + l * 4];
#pragma unroll
for (int k = 0; k < 4; k++) {
sum[k] = left[k] * right[k];
}
}