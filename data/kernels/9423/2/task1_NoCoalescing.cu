#include "includes.h"
__global__ void task1_NoCoalescing(unsigned const* a, unsigned const* b, unsigned* result, size_t size)
{
auto index = blockIdx.x * blockDim.x + threadIdx.x + 7;
if (index > size + 6) {
return;
}
if (index >= size) {
index -= 7;
}
result[index] = a[index] * b[index];
}