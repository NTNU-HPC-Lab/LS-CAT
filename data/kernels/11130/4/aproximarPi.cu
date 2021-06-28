#include "includes.h"
__global__ void aproximarPi( float *x, float *y, int *z, int tam) {
int i = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 2047
int j = threadIdx.y + blockIdx.y*blockDim.y; // 0 - 2047
int index = j + i*tam; // 0 - 4194303

if( (x[index] * x[index] + y[index] * y[index]) <= 1.0f){
atomicAdd(z, 1);
}
}