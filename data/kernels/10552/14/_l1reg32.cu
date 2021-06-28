#include "includes.h"
__global__ void _l1reg32(int n, double l1, float *w, float *dw) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
if (w[i] > 0) dw[i] += l1;
else if (w[i] < 0) dw[i] -= l1;
i += blockDim.x * gridDim.x;
}
}