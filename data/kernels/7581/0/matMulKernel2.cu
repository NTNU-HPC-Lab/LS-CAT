#include "includes.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif


// generate a random square matrix
__global__ void matMulKernel2(float* P, float* M, float* N, int width) {
__shared__ float Mds[2][2];
__shared__ float Nds[2][2];
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int row = by*2 + ty; int col = bx*2 + tx;
float pVal = 0;

for(int ph = 0; ph < width/2; ++ph) {
Mds[ty][tx] = M[row*width + ph*2 + tx];
Nds[ty][tx] = N[(ph*2 + ty)*width + col];
__syncthreads();
for(int k = 0; k < 2; ++k)
pVal += Mds[ty][k]*Nds[k][tx];
__syncthreads();
}
P[row*width + col] = pVal;
}