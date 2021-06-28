#include "includes.h"
__global__ void DataPointMap(int size, const double *inputX, const double *inputY, double *output, const double *inFreeArray, int length) {
const long ix = threadIdx.x + blockIdx.x * (long)blockDim.x;
if (ix < size) {
// copy int array
const double *inArrayBody = &inputX[ix* length];
double *outArrayBody = &output[ix* length];

for (long i = 0; i < length; i++) {
outArrayBody[i] = inArrayBody[i] + inFreeArray[i];
}
}
}