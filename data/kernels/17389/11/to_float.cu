#include "includes.h"
__global__ void to_float(float *out, int *in, int size) {
int element = threadIdx.x + blockDim.x * blockIdx.x;
if (element >= size) return;
out[element] = float(in[element]);
}