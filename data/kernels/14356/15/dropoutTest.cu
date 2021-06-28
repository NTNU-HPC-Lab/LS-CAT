#include "includes.h"
extern "C"

extern "C"
__global__ void dropoutTest( const float* arguments, float* results, const float dropoutFraction, const long size ) {
const int X = gridDim.x;
const int index = gridDim.y * X * threadIdx.x + X * blockIdx.y + blockIdx.x;

if(index < size) {
results[index] = arguments[index] * (1.f - dropoutFraction);
}
}