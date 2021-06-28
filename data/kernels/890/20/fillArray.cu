#include "includes.h"
__global__ void fillArray(int8_t *dest, int loop) {
const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
const size_t k = blockDim.x * gridDim.x;
for (int n=0; n<loop; n++) {
dest[i+n*k] = sin((i+n*k)/(float)100.0)*30;
}
}