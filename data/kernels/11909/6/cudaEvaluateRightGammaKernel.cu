#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaEvaluateRightGammaKernel(int *wptr, double *x1, double *x2, double *diagptable, double *output, const int limit) {
const int i = blockIdx.x * blockDim.x + threadIdx.x;
output[i] = 0.0;
if (i >= limit) {
return;
}
int j;
double term = 0.0;
x1 += 16 * i;
x2 += 16 * i;
#pragma unroll
for (j = 0; j < 4; j++) {
term += x1[0] * x2[0] * diagptable[0];
term += x1[1] * x2[1] * diagptable[1];
term += x1[2] * x2[2] * diagptable[2];
term += x1[3] * x2[3] * diagptable[3];
x1 += 4;
x2 += 4;
diagptable += 4;
}
term = log(0.25 * fabs(term));
output[i] += wptr[i] * term;
}