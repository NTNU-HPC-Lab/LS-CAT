#include "includes.h"
__global__ void add(int* a, int* b, int* c) {
int id = blockIdx.x * blockDim.x + threadIdx.x;
c[id] = a[id] + b[id];
}