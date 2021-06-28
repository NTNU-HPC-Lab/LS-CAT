#include "includes.h"
__global__ void _add1(int n, float val, float *x) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
x[i] += val;
i += blockDim.x * gridDim.x;
}
}