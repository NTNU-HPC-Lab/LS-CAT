#include "includes.h"
__global__ void add(int n, float a, float *x, float *y){
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < n) y[i] = a*x[i] + y[i];
}