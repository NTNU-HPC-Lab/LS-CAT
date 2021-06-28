#include "includes.h"
extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"
__global__ void fHadamard( const float* argumentsA, const float* argumentsB, float* results, const long size ) {
const int X = gridDim.x;
const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

if(index < size) {
results[index] = argumentsA[index] * argumentsB[index];
}
}