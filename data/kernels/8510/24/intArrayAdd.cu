#include "includes.h"
__global__ void intArrayAdd(int *size, const int *input, int *output, const int *inFreeArray, int *length) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < *size) {
// copy int array
const int *inArrayBody = &input[ix* *length];
int *outArrayBody = &output[ix* *length];

for (long i = 0; i < *length; i++) {
outArrayBody[i] = inArrayBody[i] + inFreeArray[i];
}
}
}