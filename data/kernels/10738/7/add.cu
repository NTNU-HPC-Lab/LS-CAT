#include "includes.h"
__global__ void add(int* a, int* b, int* c) {
// calculate global id
int id = blockIdx.x * blockDim.x + threadIdx.x;

// perform calculation
c[id] = a[id] + b[id];
}