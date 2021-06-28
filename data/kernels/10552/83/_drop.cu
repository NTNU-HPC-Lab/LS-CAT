#include "includes.h"
__global__ void _drop(int n, float *x, float *xmask, float dropout, float scale) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
if (xmask[i] < dropout) x[i] = 0;
else x[i] *= scale;
i += blockDim.x * gridDim.x;
}
}