#include "includes.h"
__global__ void MatrixMult(int *M, int *N, int *P, int width)
{
int tid, tx, ty;

tx = blockIdx.x*blockDim.x + threadIdx.x;
ty = blockIdx.y*blockDim.y + threadIdx.y;
tid = ty*width + tx;
int Pv = 0, Mv = 0, Nv = 0;

for(int i = 0; i < width; i++) {
Mv = M[ty*width+i];
Nv = N[i*width+tx];
Pv += Mv * Nv;
}

P[tid] = Pv;
}