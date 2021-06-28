#include "includes.h"
/*
* This example explains how to divide the host and
* device code into separate files using vector addition
*/
#define N 64





__global__ void addKernel(float *a,float *b) {
int idx=threadIdx.x+blockIdx.x*blockDim.x;

if(idx>=N) return;
a[idx]+=b[idx];
}