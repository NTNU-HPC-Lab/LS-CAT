#include "includes.h"
__global__ void bigstencil(int* in, int* out) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = in[i] + 2;
}