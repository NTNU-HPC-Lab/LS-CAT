#include "includes.h"
__global__ void doubleArray2floatArray(const double * doubleArray, float* floatArray, const int size) {
int i = blockDim.x*blockIdx.x + threadIdx.x;
if (i < size) {
floatArray[i] = (float) doubleArray[i];
}
}