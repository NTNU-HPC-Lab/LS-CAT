#include "includes.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif


// generate a random square matrix
__global__ void matMulKernel4(float* P, float* M, float* N, int width) {
__shared__ float Mds4[4][4];
__shared__ float Nds4[4][4];
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int row = by*4 + ty; int col = bx*4 + tx;
float pVal = 0;

for(int ph = 0; ph < width/4; ++ph) {
Mds4[ty][tx] = M[row*width + ph*4 + tx];
Nds4[ty][tx] = N[(ph*4 + ty)*width + col];
__syncthreads();
for(int k = 0; k < 4; ++k)
pVal += Mds4[ty][k]*Nds4[k][tx];
__syncthreads();
}
P[row*width + col] = pVal;
}