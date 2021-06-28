#include "includes.h"


__global__ void task2Kernel(unsigned const* a, unsigned const* b, unsigned* result, size_t size)
{
auto index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= size) {
return;
}
result[index] = a[index] * b[index];
}