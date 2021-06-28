#include "includes.h"
__global__ void load(int size, const long *in) {
const int ix = threadIdx.x + blockIdx.x * blockDim.x;

if (ix < size) {
}
}