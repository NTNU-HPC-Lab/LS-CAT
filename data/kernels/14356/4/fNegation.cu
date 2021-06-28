#include "includes.h"
extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"
__global__ void fNegation( const float* arguments, float* results, const long size ) {
const int X = gridDim.x;
const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

if(index < size) {
results[index] = -arguments[index];
}
}