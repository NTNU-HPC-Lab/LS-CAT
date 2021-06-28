#include "includes.h"
__global__ void identity(int *size, const int *input, int *output) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < *size) {
output[ix] = input[ix];
}
}