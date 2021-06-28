#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaEvaluateLeftGammaKernel(int *wptr, double *x2, double *tipVector, unsigned char *tipX1, double *diagptable, double *output, const int limit) {
const int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= limit) {
output[i] = 0.0;
return;
}
int j;
double term = 0.0;
tipVector += 4 * tipX1[i];
x2 += 16 * i;
#pragma unroll
for (j = 0; j < 4; j++) {
term += tipVector[0] * x2[0] * diagptable[0];
term += tipVector[1] * x2[1] * diagptable[1];
term += tipVector[2] * x2[2] * diagptable[2];
term += tipVector[3] * x2[3] * diagptable[3];
x2 += 4;
diagptable += 4;
}
term = log(0.25 * fabs(term));
output[i] = wptr[i] * term;
}