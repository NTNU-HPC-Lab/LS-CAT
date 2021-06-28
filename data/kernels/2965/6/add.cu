#include "includes.h"
__global__ void add(const int *a, const int *b, int *dest, const size_t length) {

for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < length;
tid += blockDim.x * gridDim.x) {
dest[tid] = a[tid] + b[tid];
}
}