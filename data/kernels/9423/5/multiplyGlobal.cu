#include "includes.h"
__global__ void multiplyGlobal(unsigned const* left, unsigned const* right, unsigned* result, size_t size)
{
auto row = blockIdx.y * blockDim.y + threadIdx.y;
auto col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < size && col < size) {
auto sum = 0u;
for (int k = 0; k < size; k++) {
sum += left[row * size + k] * right[k * size + col];
}
result[row * size + col] = sum;
}
}