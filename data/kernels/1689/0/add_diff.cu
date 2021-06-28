#include "includes.h"
__global__ void add_diff(float* a, const float* x, const float* y, const float c, int size){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if( i < size )
a[i] += c*(x[i] - y[i]);
}