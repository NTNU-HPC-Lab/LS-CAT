#include "includes.h"
__global__ void rowDiv(float* a, float* b, float* c, int M, int N){

int i = blockIdx.x*blockDim.x + threadIdx.x;
c[i] = a[i]/b[blockIdx.x];
}