#include "includes.h"
__global__ void IntDataPointIdentity(int *size, const int *inputX, const int *inputY, int *outputX, int *outputY, int *length) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < *size) {
// copy int array
const int *inArrayBody = &inputX[ix* *length];
int *outArrayBody = &outputX[ix* *length];

for (long i = 0; i < *length; i++) {
outArrayBody[i] = inArrayBody[i];
}

// copy int scalar value
outputY[ix] = inputY[ix];
}
}