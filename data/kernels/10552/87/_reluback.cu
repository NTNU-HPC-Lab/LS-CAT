#include "includes.h"
__global__ void _reluback(int n, float *y, float *dy) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
if (y[i] <= 0) dy[i] = 0;
i += blockDim.x * gridDim.x;
}
}