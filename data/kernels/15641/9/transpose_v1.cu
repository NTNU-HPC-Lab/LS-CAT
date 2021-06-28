#include "includes.h"
__global__ void transpose_v1(float* a,float* b, int n){
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;

if(i >= n || j >= n) return;

b[n*j+i] = a[n*i+j];

}