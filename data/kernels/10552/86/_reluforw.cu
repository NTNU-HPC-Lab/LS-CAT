#include "includes.h"
__global__ void _reluforw(int n, float *y) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
if (y[i] < 0) y[i] = 0;
i += blockDim.x * gridDim.x;
}
}