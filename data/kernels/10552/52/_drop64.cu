#include "includes.h"
__global__ void _drop64(int n, double *x, double *xmask, double dropout, double scale) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
if (xmask[i] < dropout) x[i] = 0;
else x[i] *= scale;
i += blockDim.x * gridDim.x;
}
}