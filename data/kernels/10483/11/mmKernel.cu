#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void mmKernel(float* popsIn, float* popsOut, float* mmm, int patches) {
int ii = threadIdx.x;

if (ii < patches) {
extern __shared__ float s[];

s[ii] = 0.0;

for (int jj = 0; jj < patches; jj++) {
s[ii] += popsIn[ii]*mmm[ii*patches + jj];
}
__syncthreads();

popsOut[ii] = s[ii];
}
}