#include "includes.h"
extern "C" {
}
__global__ void fill_u32(unsigned int *y, unsigned int elem, unsigned int len) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < len) {
y[tid] = elem;
}
}