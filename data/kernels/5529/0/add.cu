#include "includes.h"


__global__ void add(std::size_t  n, const float *x, float *y) {
std::size_t  index = blockIdx.x * blockDim.x + threadIdx.x;
std::size_t  stride = blockDim.x * gridDim.x;
for (auto i = index; i < n; i += stride) y[i] = x[i] + y[i];
}