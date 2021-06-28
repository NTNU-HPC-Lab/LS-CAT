#include "includes.h"
__global__ void floatArray2doubleArray(const float * floatArray, double* doubleArray, const int size) {
int i = blockDim.x*blockIdx.x + threadIdx.x;
if (i < size) {
doubleArray[i] = (double) floatArray[i];
}
}