#include "includes.h"
__global__ void add(const int *a, const int *b, int *dest, const size_t length) {
int tid = blockIdx.x;

if (tid < length) {
dest[tid] = a[tid] - b[tid];
}
}