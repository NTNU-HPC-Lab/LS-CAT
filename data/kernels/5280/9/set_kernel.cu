#include "includes.h"
__global__ void set_kernel(const int n, const float alpha, float *y) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);
i += blockDim.x * gridDim.x) {
y[i] = alpha;
}
}