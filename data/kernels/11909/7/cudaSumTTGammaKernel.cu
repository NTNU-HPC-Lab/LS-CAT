#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaSumTTGammaKernel(unsigned char *tipX1, unsigned char *tipX2, double *tipVector, double *sumtable, int limit) {
const int n = blockIdx.x * blockDim.x + threadIdx.x;
if (n >= limit) {
return;
}
const int i = n / 4, j = n % 4;
double *left = &(tipVector[4 * tipX1[i]]);
double *right = &(tipVector[4 * tipX2[i]]);
double *sum = &sumtable[i * 16 + j * 4];
#pragma unroll
for (int k = 0; k < 4; k++) {
sum[k] = left[k] * right[k];
}
}