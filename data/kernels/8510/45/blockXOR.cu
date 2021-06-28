#include "includes.h"
__global__ void blockXOR(int size, const char *input, char *output, long key) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix * 8 < size) {
((long *)output)[ix] = ((const long *)input)[ix] ^ key;
}
}