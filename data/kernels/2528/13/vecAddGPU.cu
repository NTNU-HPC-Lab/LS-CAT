#include "includes.h"
__global__ void vecAddGPU(double *a, double *b, double *c, double n){

int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < n){
c[id] = a[id] + b[id];
}
}