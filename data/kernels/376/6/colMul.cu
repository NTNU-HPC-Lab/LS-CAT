#include "includes.h"
__global__ void colMul(float* a, float* b, float* c, int M, int N){

int i = blockIdx.x*blockDim.x + threadIdx.x;
if(i<M){
int ind = i + blockIdx.y*M;
c[ind] = a[ind]*b[i];
}
}