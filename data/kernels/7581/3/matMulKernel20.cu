#include "includes.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif


// generate a random square matrix
__global__ void matMulKernel20(float* P, float* M, float* N, int width) {
__shared__ float Mds20[20][20];
__shared__ float Nds20[20][20];
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int row = by*20 + ty; int col = bx*20 + tx;
float pVal = 0;

for(int ph = 0; ph < width/20; ++ph) {
Mds20[ty][tx] = M[row*width + ph*20 + tx];
Nds20[ty][tx] = N[(ph*20 + ty)*width + col];
__syncthreads();
for(int k = 0; k < 20; ++k)
pVal += Mds20[ty][k]*Nds20[k][tx];
__syncthreads();
}
P[row*width + col] = pVal;
}